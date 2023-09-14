# This version try to use loss descent and timestep 4, more training sample, for debug

# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.optim as optim
import dataset as datasets
from utils.loading import load_network
from data import processing, sampler, DataLoader
import models_dbsr.dbsr.dbsrnet as dbsr_nets
import actors.dbsr_actors as dbsr_actors
from trainers import SimpleTrainer, SimpleTrainer_v2, AgentTrainer
import data.transforms as tfm
from admin.multigpu import MultiGPU
from models_dbsr.loss.image_quality_v2 import PSNR, PixelWiseError
import numpy as np
import torch
import pickle as pkl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def run(settings):
    set_seed(42)

    settings.description = 'Default settings for training DBSR models on synthetic burst dataset(NightCity) with step(6), amplify factor(4), crop size(384,384), random translation'
    settings.batch_size = 16
    settings.num_workers = 16
    settings.multi_gpu = False
    settings.print_interval = 1

    settings.crop_sz = (384, 384)
    # settings.crop_sz = (448, 448)
    settings.burst_sz = 4
    settings.downsample_factor = 4 # TODO: need to revise to 4?

    # settings.burst_transformation_params = {'max_translation': 24.0,
    #                                         'max_rotation': 1.0,
    #                                         'max_shear': 0.0,
    #                                         'max_scale': 0.0,
    #                                         'border_crop': 24}
    init_permutation = np.array([
        [0,0],
        [0,2],
        [2,2],
        [2,0]
    ])
    
    settings.burst_transformation_params = {'max_translation': 3.0,
                                        'max_rotation': 0.0,
                                        'max_shear': 0.0,
                                        'max_scale': 0.0,
                                        # 'border_crop': 24,
                                        'random_pixelshift': False,
                                        'specified_translation': init_permutation}
    burst_transformation_params_val = {'max_translation': 3.0,
                                        'max_rotation': 0.0,
                                        'max_shear': 0.0,
                                        'max_scale': 0.0,
                                        # 'border_crop': 24,
                                        'random_pixelshift': False,
                                        'specified_translation': init_permutation}
    f = open("/home/yutong/zheng/projects/dbsr_rl/DBSR/util_scripts/zurich_test_meta_infos.pkl", 'rb')
    meta_infos_val = pkl.load(f)
    f.close()
    
    settings.burst_reference_aligned = True
    settings.image_processing_params = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True, 'add_noise': True}
    image_processing_params_val = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True, 'add_noise': True, 'predefined_params': meta_infos_val}

    zurich_raw2rgb_train = datasets.ZurichRAW2RGB(split='train')
    zurich_raw2rgb_val = datasets.ZurichRAW2RGB(split='test')  

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())
    transform_val = tfm.Transform(tfm.ToTensor(normalize=True))

    data_processing_train = processing.SyntheticBurstDatabaseProcessing(settings.crop_sz, settings.burst_sz,
                                                                settings.downsample_factor,
                                                                burst_transformation_params=settings.burst_transformation_params,
                                                                transform=transform_train,
                                                                image_processing_params=settings.image_processing_params,
                                                                random_crop=True)
    data_processing_val = processing.SyntheticBurstDatabaseProcessing(settings.crop_sz, settings.burst_sz,
                                                              settings.downsample_factor,
                                                              burst_transformation_params=burst_transformation_params_val,
                                                              transform=transform_val,
                                                              image_processing_params=image_processing_params_val,
                                                              random_crop=False)

    # Train sampler and loader
    dataset_train = sampler.RandomImage([zurich_raw2rgb_train], [1],
                                        samples_per_epoch=settings.batch_size * 1300, processing=data_processing_train)
    # dataset_val = sampler.RandomImage([NightCity_val], [1],
    #                                   samples_per_epoch=settings.batch_size * 1300, processing=data_processing_val)
    dataset_val = sampler.IndexedImage(zurich_raw2rgb_val, processing=data_processing_val)

    loader_train = DataLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                              stack_dim=0, batch_size=settings.batch_size)
    loader_val = DataLoader('val', dataset_val, training=False, num_workers=settings.num_workers,
                            stack_dim=0, batch_size=settings.batch_size, epoch_interval=1) # default is also 1
    
    print("train dataset length: ", len(loader_train))
    print("val dataset length: ", len(loader_val)) 

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=0)

    objective = {'rgb': PixelWiseError(metric='l1', boundary_ignore=40), 'psnr': PSNR(boundary_ignore=40)}

    loss_weight = {'rgb': 1.0}

    # 获取encoder部分
    dbsr_net = load_network('/home/yutong/zheng/projects/dbsr_rl/DBSR/pretrained_networks/dbsr_synthetic_default.pth')

    sr_encoder = dbsr_net.encoder
    sr_merging = dbsr_net.merging
    
    actor = dbsr_actors.ActorCritic_v3(num_frames=3, hidden_size=5)
    
    # load pre_actor
    pre_actor = dbsr_actors.ActorCritic(num_frames=settings.burst_sz, num_channels=4, hidden_size=5)
    checkpoint_dict = torch.load('/mnt/samsung/zheng/downloaded_datasets/zheng_ccvl21/training_log/checkpoints/dbsr/debug_psnetv12_old/ActorCritic_ep0052.pth.tar')
    net_type = type(pre_actor).__name__
    assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'
    fields = checkpoint_dict.keys()
    ignore_fields = ['settings']
    ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])
        # Load all fields
    for key in fields:
        if key in ignore_fields:
            continue
        if key == 'net':
            pre_actor.load_state_dict(checkpoint_dict[key])
        # elif key == 'optimizer':
        #     self.optimizer.load_state_dict(checkpoint_dict[key])
        # else:
        #     setattr(self, key, checkpoint_dict[key])

    if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
        net.info = checkpoint_dict['net_info']
    
    
    # optimizer = optim.Adam(actor.parameters())

    optimizer = optim.Adam([{'params': actor.parameters(), 'lr': 1e-4}],
                           lr=2e-4)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    trainer = AgentTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler=lr_scheduler, 
                               sr_net=dbsr_net, iterations=8, reward_type='psnr',
                               discount_factor=0.99, init_permutation=None, objective_burst_num=4, pre_init_permutation=init_permutation,
                               tolerance=0.5, pre_actor=pre_actor)

    trainer.train(100, load_latest=True, fail_safe=True) # (epoch, )
