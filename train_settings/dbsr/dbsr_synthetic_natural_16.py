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
import models_dbsr.dbsr.dbsrnet as dbsr_nets
import models.deeprep.deeprepnet as deeprep_nets
import actors.dbsr_actors as dbsr_actors
from trainers import SimpleTrainer
import data.transforms as tfm
from admin.multigpu import MultiGPU
from models_dbsr.loss.image_quality_v2 import PSNR, PixelWiseError
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import numpy as np
import pickle as pkl
from accelerate import Accelerator
import torch

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
    fp16 = False
    accelerator = Accelerator(
            split_batches=True,
            mixed_precision='fp16' if fp16 else 'no',
        )
    set_seed(42)
    settings.description = 'Default settings for training DBSR models on synthetic burst dataset, with random pixel shift, trans(24), rot(1.0), burst size(8), use database function'
    settings.batch_size = 32
    settings.num_workers = 12
    settings.multi_gpu = False
    settings.print_interval = 1

    settings.crop_sz = (384, 384)
    settings.burst_sz = 16
    settings.downsample_factor = 4

    permutation = np.array([[0,0],[0,1],[0,2],[0,3],[1,3],[1,2],[1,1],[1,0],[2,0],[2,1],[2,2],[2,3],[3,3],[3,2],[3,1],[3,0]])

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
    dir_path = "/home/user/zheng//DBSR/util_scripts"
    with open(os.path.join(dir_path, 'zurich_test_meta_infos.pkl'), 'rb') as f:
        meta_infos_val = pkl.load(f)
    image_processing_params_val = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True, 'add_noise': True, \
        'predefined_params': meta_infos_val}

    zurich_raw2rgb_train = datasets.ZurichRAW2RGB(split='train')
    zurich_raw2rgb_val = datasets.ZurichRAW2RGB(split='test')

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


    # Train sampler and loader
    dataset_train = sampler.RandomImage([zurich_raw2rgb_train], [1],
                                        samples_per_epoch=settings.batch_size * 1000, processing=data_processing_train)
    dataset_val = sampler.IndexedImage(zurich_raw2rgb_val, processing=data_processing_val)
    # dataset_val_2 = sampler.IndexedImage(zurich_raw2rgb_val, processing=data_processing_val_2)
    loader_train = DataLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                              stack_dim=0, batch_size=settings.batch_size)
    loader_val = DataLoader('val', dataset_val, training=False, num_workers=settings.num_workers,
                            stack_dim=0, batch_size=settings.batch_size, epoch_interval=2)
    # loader_val_2 = DataLoader('val_2', dataset_val_2, training=False, num_workers=settings.num_workers,
    #                         stack_dim=0, batch_size=1, epoch_interval=10)
    if accelerator.is_main_process:
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
    # if settings.multi_gpu:
    #     net = MultiGPU(net, dim=0)

    objective = {'rgb': PixelWiseError(metric='l1', boundary_ignore=40), 'psnr': PSNR(boundary_ignore=40)}

    loss_weight = {'rgb': 1.0}

    actors = [net]
    actors_attr = [f"{type(actor).__name__}" for actor in enumerate(actors)]
    #############LOAD LATEST CHECKPOINT###############
    actors_type = actors_attr
    checkpoint_root_path = os.path.join(settings.env.workspace_dir, 'checkpoints', settings.project_path)
    checkpoint = None
    if os.path.exists(checkpoint_root_path) and accelerator.is_main_process:
        if len(os.listdir(checkpoint_root_path)) != 0:
                files = os.listdir(checkpoint_root_path)
                files.sort()
                cp_path = os.path.join(checkpoint_root_path, files[-1])
                checkpoint = torch.load(cp_path, map_location='cpu')
                state_dict = checkpoint['net']
                print(f"Loading latest {files[-1]} from {cp_path}")
                # print(state_dict)
                net.load_state_dict(state_dict)
                print(f"Load successfully!")
                    
    else:
        os.makedirs(checkpoint_root_path, exist_ok=True)

    # net = accelerator.prepare(net)
    loader_attributes = [{'training': loader_train.training, 'name': loader_train.name, 'epoch_interval': loader_train.epoch_interval, \
        'length': loader_train.__len__()},{'training': loader_val.training, 'name': loader_val.name, 'epoch_interval': loader_val.epoch_interval, \
        'length': loader_val.__len__()}]
    loader_train = accelerator.prepare(loader_train)
    loader_val = accelerator.prepare(loader_val)


    # for param in net.alignment_net.parameters():
    #     param.requires_grad = False


    actor = dbsr_actors.DBSRSyntheticActor(net=net, objective=objective, loss_weight=loss_weight, accelerator=accelerator)

    optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, actor.net.parameters()), 'lr': 1e-4}],
                           lr=2e-4)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer = accelerator.prepare(optimizer)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.2)
    if checkpoint is not None:
        lr_scheduler.last_epoch = checkpoint['epoch']
    lr_scheduler = accelerator.prepare(lr_scheduler)
    
    trainer = SimpleTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, accelerator=accelerator, loader_attributes=loader_attributes)
    if checkpoint is not None:
        trainer.epoch = checkpoint['epoch']
     
    
    trainer.train(200, load_latest=False, fail_safe=True)
