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
import models.dbsr.dbsrnet as dbsr_nets
import actors.dbsr_actors as dbsr_actors
from trainers import SimpleTrainer
import data.transforms as tfm
from admin.multigpu import MultiGPU
from models.loss.image_quality_v2 import PSNR, PixelWiseError
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def run(settings):
    settings.description = 'Default settings for training DBSR models on synthetic burst dataset(NightCity) with step(6), amplify factor(4), crop size(384,384), random translation'
    settings.batch_size = 16
    settings.num_workers = 8
    settings.multi_gpu = False
    settings.print_interval = 1

    settings.crop_sz = (384, 384)
    settings.burst_sz = 6
    settings.downsample_factor = 4 # TODO: need to revise to 4?
    # settings.device = torch.device("cuda:0")

    # settings.burst_transformation_params = {'max_translation': 24.0,
    #                                         'max_rotation': 1.0,
    #                                         'max_shear': 0.0,
    #                                         'max_scale': 0.0,
    #                                         'border_crop': 24}
    
    permutation = np.array([
        [0,0],
        [0,1],
        [0,2],
        [1,0],
        [1,1],
        [1,2]
    ])
    
    settings.burst_transformation_params = {'max_translation': 3.0,
                                        'max_rotation': 0.0,
                                        'max_shear': 0.0,
                                        'max_scale': 0.0,
                                        'border_crop': 24,
                                        'random_pixelshift': True,
                                        'specified_translation': permutation}
    burst_transformation_params_val = {'max_translation': 3.0,
                                        'max_rotation': 0.0,
                                        'max_shear': 0.0,
                                        'max_scale': 0.0,
                                        'border_crop': 24,
                                        'random_pixelshift': False,
                                        'specified_translation': permutation}
    
    settings.burst_reference_aligned = True
    settings.image_processing_params = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True, 'add_noise': True}

    NightCity_train = datasets.NightCity(split='train') #TODO: revise the I/O
    NightCity_val = datasets.NightCity(split='val')   

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())
    transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())

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
                                                              image_processing_params=settings.image_processing_params,
                                                              random_crop=False)

    # Train sampler and loader
    dataset_train = sampler.RandomImage([NightCity_train], [1],
                                        samples_per_epoch=settings.batch_size * 300, processing=data_processing_train)
    # dataset_val = sampler.RandomImage([NightCity_val], [1],
    #                                   samples_per_epoch=settings.batch_size * 1300, processing=data_processing_val)
    dataset_val = sampler.IndexedImage(NightCity_val, processing=data_processing_val)

    loader_train = DataLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                              stack_dim=0, batch_size=settings.batch_size)
    loader_val = DataLoader('val', dataset_val, training=False, num_workers=settings.num_workers,
                            stack_dim=0, batch_size=settings.batch_size, epoch_interval=1) # default is also 1
    
    print("train dataset length: ", len(loader_train))
    print("val dataset length: ", len(loader_val)) 

    net = dbsr_nets.dbsrnet_cvpr2021(enc_init_dim=64, enc_num_res_blocks=9, enc_out_dim=512,
                                     dec_init_conv_dim=64, dec_num_pre_res_blocks=5,
                                     dec_post_conv_dim=32, dec_num_post_res_blocks=4,
                                     upsample_factor=settings.downsample_factor * 2,
                                     offset_feat_dim=64,
                                     weight_pred_proj_dim=64,
                                     num_weight_predictor_res=3,
                                     gauss_blur_sd=1.0,
                                     icnrinit=True
                                     )

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=0)

    objective = {'rgb': PixelWiseError(metric='l1', boundary_ignore=40), 'psnr': PSNR(boundary_ignore=40)}

    loss_weight = {'rgb': 1.0}

    actor = dbsr_actors.DBSRSyntheticActor(net=net, objective=objective, loss_weight=loss_weight)

    optimizer = optim.Adam([{'params': actor.net.parameters(), 'lr': 1e-4}],
                           lr=2e-4)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)
    trainer = SimpleTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(100, load_latest=True, fail_safe=True) # (epoch, )
