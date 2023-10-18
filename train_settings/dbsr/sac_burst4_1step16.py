import torch
import torch.optim as optim
import dataset as datasets
from utils.loading import load_network
from data import processing, sampler, DataLoader
import models_dbsr.dbsr.dbsrnet as dbsr_nets
import actors.dbsr_actors as dbsr_actors
from trainers import AgentSAC
import data.transforms as tfm
from admin.multigpu import MultiGPU
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import warnings
# 忽略特定的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pickle as pkl

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True

def run(settings):
    settings.description = 'adjust 4 with pixel step 1/16 LR pixel, discount_factor: 0.99, one_step_length: 1 / 8, iterations: 10, SAC'
    settings.batch_size = 1
    settings.num_workers = 12
    settings.multi_gpu = False
    settings.print_interval = 1

    settings.crop_sz = (512, 640)
    # settings.image_size = 256
    settings.burst_sz = 4
    settings.downsample_factor = 4 # TODO: need to revise to 4?
    one_step_length = 1 / 16
    base_length = 1 / settings.downsample_factor
    buffer_size = 10000

    # settings.burst_transformation_params = {'max_translation': 24.0,
    #                                         'max_rotation': 1.0,
    #                                         'max_shear': 0.0,
    #                                         'max_scale': 0.0,
    #                                         'border_crop': 24}
    permutation = np.array([[0.,0.],[0.,2.],[2.,2.],[2.,0.]])
    
    settings.burst_transformation_params = {'max_translation': 3.0,
                                        'max_rotation': 0.0,
                                        'max_shear': 0.0,
                                        'max_scale': 0.0,
                                        # 'border_crop': 24,
                                        'random_pixelshift': False,
                                        'specified_translation': permutation}
    burst_transformation_params_val = {'max_translation': 3.0,
                                        'max_rotation': 0.0,
                                        'max_shear': 0.0,
                                        'max_scale': 0.0,
                                        # 'border_crop': 24,
                                        'random_pixelshift': False,
                                        'specified_translation': permutation}
    
    settings.burst_reference_aligned = True
    settings.image_processing_params = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True, 'add_noise': True}
    dir_path = "/home/yutong/zheng/projects/dbsr_rl/DBSR/util_scripts"
    with open(os.path.join(dir_path, 'mice_val_meta_infos.pkl'), 'rb') as f:
        meta_infos_val = pkl.load(f)
    image_processing_params_val = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True, 'add_noise': True} #, \
                                #    'predefined_params': meta_infos_val}

    zurich_raw2rgb_train = datasets.MixedMiceNIR_Dai(split='train')
    zurich_raw2rgb_val = datasets.MixedMiceNIR_Dai(split='val')  

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())
    transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True))

    data_processing_train = processing.SyntheticBurstDatabaseProcessing(settings.crop_sz, settings.burst_sz,
                                                                settings.downsample_factor,
                                                                burst_transformation_params=settings.burst_transformation_params,
                                                                transform=transform_train,
                                                                image_processing_params=settings.image_processing_params,
                                                                random_crop=False)
    data_processing_val = processing.SyntheticBurstDatabaseProcessing(settings.crop_sz, settings.burst_sz,
                                                              settings.downsample_factor,
                                                              burst_transformation_params=burst_transformation_params_val,
                                                              transform=transform_val,
                                                              image_processing_params=image_processing_params_val,
                                                              random_crop=False)

    # Train sampler and loader
    dataset_train = sampler.RandomImage([zurich_raw2rgb_train], [1],
                                        samples_per_epoch=2000, processing=data_processing_train)
    # dataset_val = sampler.RandomImage([NightCity_val], [1],
    #                                   samples_per_epoch=settings.batch_size * 1300, processing=data_processing_val)
    dataset_val = sampler.IndexedImage(zurich_raw2rgb_val, processing=data_processing_val)

    loader_train = DataLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                              stack_dim=0, batch_size=settings.batch_size)
    loader_val = DataLoader('val', dataset_val, training=False, num_workers=settings.num_workers,
                            stack_dim=0, batch_size=settings.batch_size, epoch_interval=2) # default is also 1
    
    print("train dataset length: ", len(loader_train))
    print("val dataset length: ", len(loader_val)) 

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=0)

    # 获取encoder部分
    dbsr_net = load_network('/mnt/samsung/zheng_data/training_log/checkpoints/dbsr/deeprep_synthetic_mice_4/best.pth.tar')
    
    actors = [dbsr_actors.ActorSAC(num_frames=settings.burst_sz, hidden_size=5), dbsr_actors.qValueNetwork(num_frames=settings.burst_sz), \
        dbsr_actors.qValueNetwork(num_frames=settings.burst_sz)]

    # optimizer = optim.Adam(actor.parameters())

    actor_optimizer = optim.Adam([{'params': actors[0].parameters(), 'lr': 1e-4}],
                           lr=2e-4)
    critic_1_optimizer = optim.Adam([{'params': actors[1].parameters(), 'lr': 1e-3}],
                           lr=2e-3)
    critic_2_optimizer = optim.Adam([{'params': actors[2].parameters(), 'lr': 1e-3}],
                           lr=2e-3)
    log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
    log_alpha_optimizer = optim.Adam([{'params': log_alpha, 'lr': 1e-3}],
                           lr=2e-3)

    actor_lr_scheduler = optim.lr_scheduler.MultiStepLR(actor_optimizer, milestones=[100, 150], gamma=0.2)
    critic_1_lr_scheduler = optim.lr_scheduler.MultiStepLR(critic_1_optimizer, milestones=[100, 150], gamma=0.2)
    critic_2_lr_scheduler = optim.lr_scheduler.MultiStepLR(critic_2_optimizer, milestones=[100, 150], gamma=0.2)
    log_alpha_lr_scheduler = optim.lr_scheduler.MultiStepLR(log_alpha_optimizer, milestones=[100, 150], gamma=0.2)
    
    trainer = AgentSAC(actors, 
                        [loader_train, loader_val], 
                        actor_optimizer, critic_1_optimizer, critic_2_optimizer, log_alpha_optimizer,
                        settings, 
                        actor_lr_scheduler=actor_lr_scheduler, 
                        critic_1_lr_scheduler=critic_1_lr_scheduler, 
                        critic_2_lr_scheduler=critic_2_lr_scheduler, 
                        log_alpha_lr_scheduler=log_alpha_lr_scheduler,
                        log_alpha=log_alpha, 
                        sr_net=dbsr_net, iterations=10, reward_type='psnr',
                        discount_factor=0.98, init_permutation=permutation, one_step_length=one_step_length, base_length=base_length, minimal_size=500)

    trainer.train(200, load_latest=True, fail_safe=True, buffer_size=buffer_size) # (epoch, )
