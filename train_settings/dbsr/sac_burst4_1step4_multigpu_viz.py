import torch
import torch.nn as nn
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import pickle as pkl
from actors.dbsr_actors import qValueNetwork
from accelerate import Accelerator, DistributedType

def run(settings):
    fp16 = False
    accelerator = Accelerator(
            split_batches=True,
            mixed_precision='fp16' if fp16 else 'no'
        )

    ##############SETTINGS#####################
    settings.description = 'adjust 4 with pixel step 1/8 LR pixel, discount_factor: 0.99, one_step_length: 1 / 8, iterations: 10, SAC'
    settings.batch_size = 1
    sample_size = 16
    settings.num_workers = 32
    settings.multi_gpu = False
    settings.print_interval = 1
    used_weights_for_validate_traj = True
    weigths_path = "/mnt/7T/zheng/DBSR_results/checkpoints/dbsr/sac_burst4_1step4_multigpu/ActorSAC_0/ep0173.pth.tar"
    settings.crop_sz = (512, 640)
    # settings.crop_sz = (512, 640)
    settings.burst_sz = 4
    settings.downsample_factor = 4
    one_step_length = 1 / 4
    base_length = 1 / settings.downsample_factor
    buffer_size = 10000
    

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
    dir_path = "/home/user/zheng/DBSR/util_scripts"
    with open(os.path.join(dir_path, 'mice_val_meta_infos.pkl'), 'rb') as f:
        meta_infos_val = pkl.load(f)
    image_processing_params_val = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True, 'add_noise': True, \
                                        'predefined_params': meta_infos_val}

    ################DEFINE DATALOADER################
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
                                                              random_crop=False, return_rgb_busrt=True)

    # Train sampler and loader
    dataset_train = sampler.RandomImage([zurich_raw2rgb_train], [1],
                                        samples_per_epoch=settings.batch_size * 100, processing=data_processing_train)
    dataset_val = sampler.IndexedImage(zurich_raw2rgb_val, processing=data_processing_val)

    loader_train = DataLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                              stack_dim=0, pin_memory=True, batch_size=settings.batch_size)
    loader_val = DataLoader('val', dataset_val, training=False, num_workers=settings.num_workers,
                            stack_dim=0, pin_memory=True, batch_size=settings.batch_size, epoch_interval=1) # default is also 1
    if accelerator.is_main_process:
        print("train dataset length: ", len(loader_train))
        print("val dataset length: ", len(loader_val)) 

    #############IMPORT SR NET#####################
    sr_net = load_network('/home/user/zheng/DBSR/pretrained_networks/best.pth.tar')

    #############DEFINE SAC 1-ACTOR 4-CRITICS###########
    p_net  = dbsr_actors.ActorSAC(num_frames=settings.burst_sz, hidden_size=5)
    
    q_net1 = dbsr_actors.qValueNetwork(num_frames=settings.burst_sz)
    q_net2 = dbsr_actors.qValueNetwork(num_frames=settings.burst_sz)

    target_critic_1 = qValueNetwork(num_frames=settings.burst_sz)
    target_critic_2 = qValueNetwork(num_frames=settings.burst_sz)

    actors = [p_net, q_net1, q_net2, target_critic_1, target_critic_2]
    actors_attr = [f"{type(actor).__name__}_{idx}" for idx, actor in enumerate(actors)]
    actors_attr = actors_attr[:-2]
    #############LOAD LATEST CHECKPOINT###############
    actors_type = actors_attr
    checkpoint_root_path = os.path.join(settings.env.workspace_dir, 'checkpoints', settings.project_path)
    checkpoint_sample_path = os.path.join(checkpoint_root_path, actors_type[0])
    if os.path.exists(checkpoint_root_path) and accelerator.is_main_process:
        if os.path.exists(checkpoint_sample_path):
            if len(os.listdir(checkpoint_sample_path)) != 0:
                nets_checkpoints_dir_path = [os.path.join(checkpoint_root_path, i) for i in actors_type]
                for idx, net_checkpoints_dir_path in enumerate(nets_checkpoints_dir_path):
                    files = os.listdir(net_checkpoints_dir_path)
                    files.sort()
                    cp_path = os.path.join(net_checkpoints_dir_path, files[-1])
                    checkpoint = torch.load(cp_path, map_location='cpu')
                    state_dict = checkpoint['net']
                    print(f"Loading latest {files[-1]} from {net_checkpoints_dir_path}")
                    # print(state_dict)
                    actors[idx].load_state_dict(state_dict)
                    print(f"Load successfully!")
    elif used_weights_for_validate_traj:
        checkpoint = torch.load(weigths_path, map_location='cpu')
        state_dict = checkpoint['net']
        actors[0].load_state_dict(state_dict)
    else:
        os.makedirs(checkpoint_root_path, exist_ok=True)

    target_critic_1.load_state_dict(q_net1.state_dict())
    target_critic_2.load_state_dict(q_net2.state_dict())
    ############DEFINE MULTIGPU SETTINGS###########
    
    sr_net = accelerator.prepare(sr_net)
    actors[0] = accelerator.prepare(actors[0])
    actors[1] = accelerator.prepare(actors[1])
    actors[2] = accelerator.prepare(actors[2])
    actors[3] = accelerator.prepare(actors[3])
    actors[4] = accelerator.prepare(actors[4])
    log_alpha = torch.tensor(np.log(1), dtype=torch.float)
    log_alpha.requires_grad = False
    log_alpha = accelerator.prepare(log_alpha)

    ##############DEFINE OPTIMIZER##########
    actor_optimizer = optim.Adam(p_net.parameters(), lr=1e-4)
    # torch.save(actor_optimizer, './test.pth.tar')
    critic_1_optimizer = optim.Adam(q_net1.parameters(), lr=3e-4)
    critic_2_optimizer = optim.Adam(q_net2.parameters(), lr=3e-4)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

    if os.path.exists(checkpoint_root_path) and accelerator.is_main_process:
        if os.path.exists(checkpoint_sample_path):
            if len(os.listdir(checkpoint_sample_path)) != 0:
                net_checkpoints_dir_path = os.path.join(checkpoint_root_path, actors_type[0])
                files = os.listdir(net_checkpoints_dir_path)
                files.sort()
                cp_path = os.path.join(net_checkpoints_dir_path, files[-1])
                checkpoint = torch.load(cp_path, map_location='cpu')
                actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
                critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer'])
                log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])

    actor_optimizer, critic_1_optimizer, critic_2_optimizer, log_alpha_optimizer = accelerator.prepare(actor_optimizer, critic_1_optimizer, critic_2_optimizer, log_alpha_optimizer)

    ############DEFINE LRSCHEDULER#############
    actor_lr_scheduler = optim.lr_scheduler.MultiStepLR(actor_optimizer, milestones=[100, 150], gamma=0.2)
    critic_1_lr_scheduler = optim.lr_scheduler.MultiStepLR(critic_1_optimizer, milestones=[100, 150], gamma=0.2)
    critic_2_lr_scheduler = optim.lr_scheduler.MultiStepLR(critic_2_optimizer, milestones=[100, 150], gamma=0.2)
    log_alpha_lr_scheduler = optim.lr_scheduler.MultiStepLR(log_alpha_optimizer, milestones=[100, 150], gamma=0.2)

    if os.path.exists(checkpoint_root_path) and accelerator.is_main_process:
        if os.path.exists(checkpoint_sample_path):
            if len(os.listdir(checkpoint_sample_path)) != 0:
                net_checkpoints_dir_path = os.path.join(checkpoint_root_path, actors_type[0])
                files = os.listdir(net_checkpoints_dir_path)
                files.sort()
                cp_path = os.path.join(net_checkpoints_dir_path, files[-1])
                checkpoint = torch.load(cp_path, map_location='cpu')
                actor_lr_scheduler.last_epoch = checkpoint['epoch']
                critic_1_lr_scheduler.last_epoch = checkpoint['epoch']
                critic_2_lr_scheduler.last_epoch = checkpoint['epoch']
                log_alpha_lr_scheduler.last_epoch = checkpoint['epoch']

    actor_lr_scheduler = accelerator.prepare(actor_lr_scheduler)
    critic_1_lr_scheduler = accelerator.prepare(critic_1_lr_scheduler)
    critic_2_lr_scheduler = accelerator.prepare(critic_2_lr_scheduler)
    log_alpha_lr_scheduler = accelerator.prepare(log_alpha_lr_scheduler)

    ###########DEFINE LOADER################
    loader_attributes = [{'training': loader_val.training, 'name': loader_val.name, 'epoch_interval': loader_val.epoch_interval, \
        'length': loader_val.__len__()}]
    loader_train = accelerator.prepare(loader_train)
    loader_val = accelerator.prepare(loader_val)

    ############DEFINE TRAINER###############
    trainer = AgentSAC(actors, 
                        [loader_val], 
                        actor_optimizer, critic_1_optimizer, critic_2_optimizer, log_alpha_optimizer,
                        settings, 
                        actor_lr_scheduler=actor_lr_scheduler, 
                        critic_1_lr_scheduler=critic_1_lr_scheduler, 
                        critic_2_lr_scheduler=critic_2_lr_scheduler, 
                        log_alpha_lr_scheduler=log_alpha_lr_scheduler,
                        log_alpha=log_alpha, 
                        sr_net=sr_net, iterations=2, reward_type='psnr',
                        discount_factor=0.99, init_permutation=permutation, one_step_length=one_step_length, base_length=base_length,
                        sample_size=sample_size, accelerator=accelerator,
                        loader_attributes=loader_attributes,
                        actors_attr=actors_attr, target_entropy=-5, minimal_size=200, gpus_num=8,
                        save_results=True, saving_dir="/mnt/7T/zheng/DBSR_results/loggings/b4_1-8_20231022")

    trainer.train(200, load_latest=False, fail_safe=True, buffer_size=buffer_size) # (epoch, )
