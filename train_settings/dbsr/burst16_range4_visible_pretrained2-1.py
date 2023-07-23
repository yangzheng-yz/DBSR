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
import models.dbsr.dbsrnet as dbsr_nets
import actors.dbsr_actors as dbsr_actors
from trainers import SimpleTrainer
import data.transforms as tfm
from admin.multigpu import MultiGPU
from models.loss.image_quality_v2 import PSNR, PixelWiseError
import os
import pickle as pkl
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run(settings):
    settings.description = 'Default settings for training DBSR models on real nir visible dataset, range(4), burst size(16), use database function'
    settings.batch_size = 7
    settings.num_workers = 8
    settings.multi_gpu = False
    settings.print_interval = 1

    settings.crop_sz = (448, 448)
    settings.burst_sz = 16
    settings.downsample_factor = 4

    # f=open("/home/yutong/zheng/projects/dbsr_us/util_scripts/traj_files/zurich_trajectory_step-4_range-4.pkl", 'rb')
    # permutations = pkl.load(f)
    # permutation = np.array([[0,0],[0,1],[0,2],[0,3],
    #                         [1,0],[1,1],[1,2],[1,3],
    #                         [2,0],[2,1],[2,2],[2,3],
    #                         [3,0],[3,1],[3,2],[3,3]]) # VISIBLE dataset do not need synthetic shifts
    # f.close()

    # settings.burst_transformation_params = {'max_translation': 24.0,
    #                                         'max_rotation': 1.0,
    #                                         'max_shear': 0.0,
    #                                         'max_scale': 0.0,
    #                                         # 'border_crop': 0,
    #                                         'random_pixelshift': False,
    #                                         'specified_translation': permutation
    #                                         }
    
    # burst_transformation_params_val = {'max_translation': 4.0,
    #                                     'max_rotation': 0.0,
    #                                     'max_shear': 0.0,
    #                                     'max_scale': 0.0,
    #                                     # 'border_crop': 0,
    #                                     'random_pixelshift': False,
    #                                     'specified_translation': permutation
    #                                     }
    
    settings.burst_reference_aligned = True
    settings.image_processing_params = {'random_ccm': False, 'random_gains': False, 'smoothstep': False, 'gamma': False, 'add_noise': True}

    nir_visible_train = datasets.nir_visible(burst_sz=settings.burst_sz, split='train-1')
    nir_visible_val = datasets.nir_visible(burst_sz=settings.burst_sz, split='test-1')

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True))
    transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True))

    data_processing_train = processing.VisibleBurstProcessing(settings.crop_sz,
                                                                settings.downsample_factor,
                                                                # burst_transformation_params=settings.burst_transformation_params,
                                                                transform=transform_train,
                                                                image_processing_params=settings.image_processing_params,
                                                                random_crop=False,
                                                                random_flip=True)
    data_processing_val = processing.VisibleBurstProcessing(settings.crop_sz,
                                                              settings.downsample_factor,
                                                            #   burst_transformation_params=burst_transformation_params_val,
                                                              transform=transform_val,
                                                              image_processing_params=settings.image_processing_params,
                                                              random_crop=False,
                                                              random_flip=False)

    # Train sampler and loader
    dataset_train = sampler.RandomImage([nir_visible_train], [1],
                                        samples_per_epoch=settings.batch_size * 1000, processing=data_processing_train)
    dataset_val = sampler.IndexedImage(nir_visible_val, processing=data_processing_val)


    loader_train = DataLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                              stack_dim=0, batch_size=settings.batch_size)
    loader_val = DataLoader('val', dataset_val, training=False, num_workers=settings.num_workers,
                            stack_dim=0, batch_size=settings.batch_size, epoch_interval=1)

    net = load_network('/mnt/data0/zheng/training_results/checkpoints/dbsr/burst16_range4_visible_pretrained2/DBSRNet_ep0151.pth.tar')
    
    # net = dbsr_nets.dbsrnet_cvpr2021(enc_init_dim=64, enc_num_res_blocks=9, enc_out_dim=512,
    #                                  dec_init_conv_dim=64, dec_num_pre_res_blocks=5,
    #                                  dec_post_conv_dim=32, dec_num_post_res_blocks=4,
    #                                  upsample_factor=settings.downsample_factor * 2,
    #                                  offset_feat_dim=64,
    #                                  weight_pred_proj_dim=64,
    #                                  num_weight_predictor_res=3,
    #                                  gauss_blur_sd=1.0,
    #                                  icnrinit=True
    #                                  )

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=0)

    objective = {'rgb': PixelWiseError(metric='l1', boundary_ignore=40), 'psnr': PSNR(boundary_ignore=40)} # , 'perceptual': PixelWiseError(metric='perceptual', boundary_ignore=40)}

    loss_weight = {'rgb': 1.0} # , 'perceptual': 0.1}

    actor = dbsr_actors.DBSRSyntheticActor(net=net, objective=objective, loss_weight=loss_weight)

    optimizer = optim.Adam([{'params': actor.net.parameters(), 'lr': 1e-4}],
                           lr=2e-4)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)
    trainer = SimpleTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(200, load_latest=True, fail_safe=True)
