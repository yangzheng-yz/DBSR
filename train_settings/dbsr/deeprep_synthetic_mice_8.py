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
from data import processing, sampler, DataLoader
# import models_dbsr.dbsr.dbsrnet as dbsr_nets
import models.deeprep.deeprepnet as deeprep_nets
import actors.dbsr_actors as dbsr_actors
from trainers import SimpleTrainer
import data.transforms as tfm
from admin.multigpu import MultiGPU
from models_dbsr.loss.image_quality_v2 import PSNR, PixelWiseError
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np

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
    settings.description = 'Default settings for training DBSR models on synthetic burst dataset, with random pixel shift, trans(24), rot(1.0), burst size(8), use database function'
    settings.batch_size = 3
    settings.num_workers = 8
    settings.multi_gpu = False
    settings.print_interval = 1

    settings.crop_sz = (512, 640)
    settings.burst_sz = 8
    settings.downsample_factor = 4

    permutation = np.array([[0,0],[1,0],[2,0],[3,0],[3,3],[0,1],[3,1],[0,3]])

    settings.burst_transformation_params = {'max_translation': 24.0,
                                            'max_rotation': 1.0,
                                            'max_shear': 0.0,
                                            'max_scale': 0.0,
                                            # 'border_crop': 0,
                                            'random_pixelshift': True,
                                            'specified_translation': permutation
                                            }
    burst_transformation_params_val = {
                                        # 'max_translation': 4.0,
    #                                     'max_rotation': 0.0,
    #                                     'max_shear': 0.0,
    #                                     'max_scale': 0.0,
    #                                     'border_crop': 4,
                                        'random_pixelshift': False,
                                        'specified_translation': permutation
                                        }
    
    settings.burst_reference_aligned = True
    settings.image_processing_params = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True, 'add_noise': True}
    image_processing_params_val = {'random_ccm': False, 'random_gains': False, 'smoothstep': False, 'gamma': False, 'add_noise': False}

    zurich_raw2rgb_train = datasets.MixedMiceNIR_Dai(split='train')
    zurich_raw2rgb_val = datasets.MixedMiceNIR_Dai(split='val')

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())
    transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True))

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
    data_processing_val_2 = processing.SyntheticBurstDatabaseProcessing(settings.crop_sz, settings.burst_sz,
                                                              1.0,
                                                              burst_transformation_params=burst_transformation_params_val,
                                                              transform=transform_val,
                                                              image_processing_params=image_processing_params_val,
                                                              random_crop=False)

    # Train sampler and loader
    dataset_train = sampler.RandomImage([zurich_raw2rgb_train], [1],
                                        samples_per_epoch=settings.batch_size * 1000, processing=data_processing_train)
    dataset_val = sampler.IndexedImage(zurich_raw2rgb_val, processing=data_processing_val)
    dataset_val_2 = sampler.IndexedImage(zurich_raw2rgb_val, processing=data_processing_val_2)
    loader_train = DataLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                              stack_dim=0, batch_size=settings.batch_size)
    loader_val = DataLoader('val', dataset_val, training=False, num_workers=settings.num_workers,
                            stack_dim=0, batch_size=settings.batch_size, epoch_interval=5)
    loader_val_2 = DataLoader('val_2', dataset_val_2, training=False, num_workers=settings.num_workers,
                            stack_dim=0, batch_size=1, epoch_interval=10)

    net = deeprep_nets.deeprep_sr_iccv21(num_iter=3, enc_dim=64, enc_num_res_blocks=5, enc_out_dim=256,
                                         dec_dim_pre=64, dec_dim_post=32, dec_num_pre_res_blocks=5,
                                         dec_num_post_res_blocks=5,
                                         dec_in_dim=64, dec_upsample_factor=4, gauss_blur_sd=1,
                                         feature_degradation_upsample_factor=2, use_feature_regularization=False,
                                         wp_ref_offset_noise=0.00)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=0)

    objective = {'rgb': PixelWiseError(metric='l1', boundary_ignore=40), 'psnr': PSNR(boundary_ignore=40)}

    loss_weight = {'rgb': 1.0}

    actor = dbsr_actors.DBSRSyntheticActor(net=net, objective=objective, loss_weight=loss_weight)

    optimizer = optim.Adam([{'params': actor.net.parameters(), 'lr': 1e-4}],
                           lr=2e-4)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.2)
    trainer = SimpleTrainer(actor, [loader_train, loader_val_2, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(200, load_latest=True, fail_safe=True)