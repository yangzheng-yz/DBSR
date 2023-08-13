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

import os
from collections import OrderedDict
from trainers.base_trainer import BaseTrainer
from admin.stats import AverageMeter, StatValue
from admin.tensorboard import TensorboardWriter
import torch
import time
import numpy as np
from models.loss.image_quality_v2 import PSNR, PixelWiseError, SSIM
import data.camera_pipeline as rgb2raw
import data.synthetic_burst_generation as syn_burst_generation



class SimpleTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
        self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        for i, data in enumerate(loader, 1):
            # get inputs
            if self.move_data_to_gpu:
                # print("!!!!!!!!!!!!!!!!!!!!data's device: ", self.device)
                data = data.to(self.device)
            # print("ADSSDSDSDSDS: ", data['burst'].size()) # (batch_size, burst_size, channels, height, width)
            # print("!!!!!!gt size: ", data['frame_gt'].size()) # (batch_size, channels, height, width)
            
            data['epoch'] = self.epoch
            data['settings'] = self.settings

            # forward pass
            loss, stats = self.actor(data)
            print("!!!!!!!!!!!loss: ", loss)
            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # update statistics
            # batch_size = data['train_images'].shape[loader.stack_dim]
            batch_size = self.settings.batch_size
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)
            print(print_str[:-5])

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                lr_list = self.lr_scheduler.get_lr()
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.module_name, self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)

class SimpleTrainer_v2(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, 
                 init_permutation=None, discount_factor=0.99, sr_net=None, 
                 lr_scheduler=None, iterations=15, 
                 interpolation_type='bilinear', reward_type='psnr'):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
        self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        
        self.iterations = iterations
        
        self.downsample_factor = settings.downsample_factor
        
        assert sr_net is not None, "You must specify a pretrained SR model to calculate reward"
        self.sr_net = sr_net
        self.sr_net = self.sr_net.to(self.device)
        
        self.interpolation_type = interpolation_type
        
        self.discount_factor = discount_factor
        
        self.init_permutation = init_permutation
        
        self.reward_type = reward_type
        
    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def _sample_actions(self, action_pdf):
        """Sample actions from the action probability distribution."""
        batch_size, num_images, num_actions = action_pdf.shape
        actions = torch.multinomial(action_pdf.view(-1, num_actions), 1)
        actions = actions.view(batch_size, num_images)
        return actions

    def _sample_actions_v2(self, action_pdf):
        """Sample actions from the action probability distribution."""
        batch_size, burst_size_1, num_actions = action_pdf.shape
        actions = torch.multinomial(action_pdf.view(-1, num_actions), 1)
        actions = actions.view(batch_size, burst_size_1)
        return actions

    # def _update_permutations(self, permutations, actions):
    #     """Update permutations based on the actions."""
    #     batch_size, num_images, _ = permutations.shape
    #     action_offsets = torch.tensor([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])
    #     for i in range(1, num_images):  # start from 1 because the base frame does not move
    #         permutations[:, i][0] = permutations[:, i] + action_offsets[actions[:, i-1]]
            
    #     return permutations
    
    def _update_permutations(self, permutations, actions):
        """Update permutations based on the actions."""
        batch_size, num_images, _ = permutations.shape
        action_offsets = torch.tensor([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])
        for i in range(1, num_images):  # start from 1 because the base frame does not move
            permutations[:, i] = permutations[:, i] + action_offsets[actions[:, i-1]]
        
        # Clip the values between 0 and 3 using modulo and loop
        while torch.any(permutations < 0):
            permutations[permutations < 0] += 4
        while torch.any(permutations > 3):
            permutations[permutations > 3] -= 4
                
        return permutations
    
    def _update_selections(self, selected_indices, actions):
        """Update permutations based on the actions."""
        batch_size, burst_size= selected_indices.shape
        action_offsets = torch.tensor([0,1,-1])
        for i in range(1, burst_size):  # start from 1 because the base frame does not move
            selected_indices[:, i] = selected_indices[:, i] + action_offsets[actions[:, i-1]]
        
        # # Clip the values between 0 and 3 using modulo and loop
        # while torch.any(selected_indices[:,[1,2,3]] < 1):
        #     selected_indices[:,[1,2,3]][selected_indices[:,[1,2,3]] < 1] = 1
        # while torch.any(selected_indices[:,[1,2,3]] > 15):
        #     selected_indices[:,[1,2,3]][selected_indices[:,[1,2,3]] > 15] = 15
                
        return selected_indices

    def _calculate_reward(self, frame_gt, pred_current, pred_last, reward_func=None):
        """Calculate the reward as the difference of PSNR between current and last prediction."""
        assert reward_func is not None, "You must specify a reward function."
        if self.reward_type == 'psnr':
            metric_current = reward_func[self.reward_type](pred_current, frame_gt)
            metric_last = reward_func[self.reward_type](pred_last, frame_gt)
            reward = metric_current - metric_last
        elif self.reward_type == 'ssim':
            metric_current = reward_func[self.reward_type](pred_current, frame_gt)
            metric_last = reward_func[self.reward_type](pred_last, frame_gt)
            reward = 10*(metric_current - metric_last)    
        print('last: ', metric_last.item())
        print('current: ', metric_current.item())
        return reward # Tensor

    def _apply_actions(self, images, permutations, downsample_factor):
        """Apply actions to a batch of images."""
        device = images.device
        # images = images.cpu()
        batch_size = images.size(0)
        burst_size = permutations.size(1)
        transformed_images = []
        for i in range(batch_size):
            image = images[i].cpu()
            # print("image type: ", image.device)
            # time.sleep(1000)
            burst_transformation_params = {'max_translation': 24.0,
                                        'max_rotation': 1.0,
                                        'max_shear': 0.0,
                                        'max_scale': 0.0,
                                        'random_pixelshift': False,
                                        'specified_translation': permutations[i]}
            image_burst_rgb, _ = syn_burst_generation.single2lrburstdatabase(image, burst_size=burst_size,
                                                        downsample_factor=self.downsample_factor,
                                                        transformation_params=burst_transformation_params,
                                                        interpolation_type=self.interpolation_type)
            image_burst = rgb2raw.mosaic(image_burst_rgb.clone())
            transformed_images.append(image_burst)
            transformed_images_stacked = torch.stack(transformed_images).to(device)
        return transformed_images_stacked


    # def cycle_dataset(self, loader):
    #     """Do a cycle of training or validation."""

    #     self.actor.train(loader.training)
    #     torch.set_grad_enabled(loader.training)

    #     self._init_timing()

    #     discount_factor = self.discount_factor # set your discount factor

    #     for i, data in enumerate(loader, 1):
    #         # print("data type: ", data.keys())
    #         # time.sleep(1000)
    #         # get inputs
    #         if self.move_data_to_gpu:
    #             data = data.to(self.device)
    #             # data = {k: v.to(self.device) for k, v in data.items()}
    #         # print("After data load Memory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")
    #         # print("Memory Cached:", torch.cuda.memory_cached() / (1024 ** 2), "MB")
    #         data['epoch'] = self.epoch
    #         data['settings'] = self.settings

    #         batch_size = data['frame_gt'].size(0)

    #         if self.init_permutation is not None:
    #             permutations = torch.tensor(self.init_permutation).repeat(batch_size, 1, 1)
    #         else:
    #             permutations = torch.tensor(np.array([[0,0],
    #                                         [0,0],
    #                                         [0,0],
    #                                         [0,0]])).repeat(batch_size, 1, 1)

    #         rewards = []
    #         log_probs = []
    #         preds = []

    #         pred, _ = self.sr_net(data['burst'])
    #         preds.append(pred)
    #         if self.reward_type == 'psnr':
    #             reward_func = {'psnr': PSNR(boundary_ignore=40)}
    #         elif self.reward_type == 'ssim':
    #             reward_func = {'ssim': SSIM(boundary_ignore=40)}
    #         else:
    #             assert 0 == 1, "wrong reward type"

    #         for it in range(self.iterations):
    #             # forward pass, to produce action_pdf
    #             actions_pdf = self.actor(data)
                
    #             # sample and apply actions
    #             actions = self._sample_actions(actions_pdf)
    #             permutations = self._update_permutations(permutations, actions)
    #             data['burst'] = self._apply_actions(data['frame_gt'], permutations, downsample_factor=self.downsample_factor)
    #             # print("After _apply_actions Memory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")

    #             # updates preds and calculate reward
    #             with torch.no_grad():
    #                 pred, _ = self.sr_net(data['burst'])
    #             # print("After updates preds Memory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")

    #             preds.append(pred)
    #             # print("!@#length preds: ", len(preds))
    #             # time.sleep(1000)
    #             reward = self._calculate_reward(data['frame_gt'], preds[-1], preds[-2], reward_func=reward_func)
    #             rewards.append(reward)

    #             # calculate log probabilities of the sampled actions
    #             log_prob = torch.log(actions_pdf.gather(2, actions.unsqueeze(-1)).squeeze(-1))
    #             log_probs.append(log_prob)
    #             # print("After calculate log probabilities Memory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")
    #             print("%s iteration permutation: " % it, permutations)

    #         # calculate discounted reward
    #         reward_iter = sum((discount_factor ** i) * reward for i, reward in enumerate(rewards))
    #         reward_normalized = reward_iter / float(self.iterations)
    #         # print("After calculate discounted reward Memory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")

    #         # calculate loss
    #         log_probs = torch.stack(log_probs)
    #         # loss_iter = -(log_probs * reward_normalized).mean()
    #         loss_iter = -(log_probs * reward_iter).sum()
    #         # print("After calculate loss Memory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")

    #         # calculate metric for the initial and final burst
    #         metric_initial = reward_func[self.reward_type](pred, data['frame_gt'])
    #         metric_final = reward_func[self.reward_type](preds[-1], data['frame_gt'])
    #         # print("After calculate PSNR Memory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")


    #         # backward pass and update weights
    #         if loader.training:
    #             self.optimizer.zero_grad()
    #             (loss_iter).backward()
    #             self.optimizer.step()

    #         # update statistics
    #         batch_size = self.settings.batch_size
    #         self._update_stats({'Loss/total': loss_iter.item(), ('%s/initial' % self.reward_type): metric_initial.item(), ('%s/final' % self.reward_type): metric_final.item(), "Improvement": metric_final.item()-metric_initial.item()}, batch_size, loader)

    #         # print statistics
    #         self._print_stats(i, loader, batch_size)
    #         # print("After backward pass Memory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")

    #         del data
    #         if "data" in locals():
    #             print("`data` has not been deleted!")

    #         del permutations
    #         if "permutations" in locals():
    #             print("`permutations` has not been deleted!")

    #         del rewards
    #         if "rewards" in locals():
    #             print("`rewards` has not been deleted!")
    #         del log_probs
    #         if "log_probs" in locals():
    #             print("`log_probs` has not been deleted!")
    #         del preds
    #         if "preds" in locals():
    #             print("`preds` has not been deleted!")            
 
    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)
        torch.autograd.set_detect_anomaly(True)

        self._init_timing()

        discount_factor = self.discount_factor # set your discount factor

        for i, data in enumerate(loader, 1):
            # print("data type: ", data.keys())
            # time.sleep(1000)
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)
                # data = {k: v.to(self.device) for k, v in data.items()}
            # print("After data load Memory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")
            # print("Memory Cached:", torch.cuda.memory_cached() / (1024 ** 2), "MB")
            data['epoch'] = self.epoch
            data['settings'] = self.settings

            batch_size = data['frame_gt'].size(0)
            selected_indices = [0, 2, 8, 10]
            selected_data = data['burst'][:, selected_indices].clone()
            selected_indices = torch.tensor(np.array([0, 2, 8, 10])).repeat(batch_size, 1)
            # selected_data = data['burst'][selected_indices].clone()
            rewards = []
            log_probs = []
            preds = []
            print("selected data size: ", selected_data.dim())
            # time.sleep(1000)
            pred, _ = self.sr_net(selected_data)
            preds.append(pred.clone())
            if self.reward_type == 'psnr':
                reward_func = {'psnr': PSNR(boundary_ignore=40)}
            elif self.reward_type == 'ssim':
                reward_func = {'ssim': SSIM(boundary_ignore=40)}
            else:
                assert 0 == 1, "wrong reward type"

            for it in range(self.iterations):
                # forward pass, to produce action_pdf
                actions_pdf = self.actor(selected_data)
                
                # sample and apply actions
                actions = self._sample_actions_v2(actions_pdf)
                # permutations = self._update_permutations(permutations, actions)
                selected_indices = self._update_selections(selected_indices, actions)
                print("%s iteration selected indices: " % it, selected_indices)
                # data['burst'] = self._apply_actions(data['frame_gt'], permutations, downsample_factor=self.downsample_factor)
                batch_indices = torch.arange(batch_size)[:, None]
                selected_data = data['burst'][batch_indices, selected_indices].clone()
                # print("%s iteration selected data dim: " % it, selected_data.dim())

                # print("After _apply_actions Memory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")

                # updates preds and calculate reward
                with torch.no_grad():
                    # pred, _ = self.sr_net(data['burst'])
                    pred, _ = self.sr_net(selected_data)
                # print("After updates preds Memory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")

                preds.append(pred.clone())
                # print("!@#length preds: ", len(preds))
                # time.sleep(1000)
                reward = self._calculate_reward(data['frame_gt'], preds[-1], preds[-2], reward_func=reward_func)
                print("outside last: ", reward_func[self.reward_type](preds[-2], data['frame_gt']))
                print("outside current: ", reward_func[self.reward_type](preds[-1], data['frame_gt']))
                rewards.append(reward)

                # calculate log probabilities of the sampled actions
                log_prob = torch.log(actions_pdf.gather(2, actions.unsqueeze(-1)).squeeze(-1))
                log_probs.append(log_prob)
                # print("After calculate log probabilities Memory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")
                # print("%s iteration permutation: " % it, permutations)
                

            # calculate discounted reward
            reward_iter = sum((discount_factor ** i) * reward for i, reward in enumerate(rewards))
            reward_normalized = reward_iter / float(self.iterations)
            # print("After calculate discounted reward Memory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")

            # calculate loss
            log_probs = torch.stack(log_probs)
            # loss_iter = -(log_probs * reward_normalized).mean()
            loss_iter = -(log_probs * reward_iter).sum()
            # print("After calculate loss Memory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")

            # calculate metric for the initial and final burst
            metric_initial = reward_func[self.reward_type](pred, data['frame_gt'])
            metric_final = reward_func[self.reward_type](preds[-1], data['frame_gt'])
            # print("After calculate PSNR Memory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")


            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                (loss_iter).backward()
                self.optimizer.step()

            # update statistics
            batch_size = self.settings.batch_size
            self._update_stats({'Loss/total': loss_iter.item(), ('%s/initial' % self.reward_type): metric_initial.item(), ('%s/final' % self.reward_type): metric_final.item(), "Improvement": metric_final.item()-metric_initial.item()}, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)
            # print("After backward pass Memory Allocated:", torch.cuda.memory_allocated() / (1024 ** 2), "MB")

            # del data
            # if "data" in locals():
            #     print("`data` has not been deleted!")

            # del permutations
            # if "permutations" in locals():
            #     print("`permutations` has not been deleted!")

            # del rewards
            # if "rewards" in locals():
            #     print("`rewards` has not been deleted!")
            # del log_probs
            # if "log_probs" in locals():
            #     print("`log_probs` has not been deleted!")
            # del preds
            # if "preds" in locals():
            #     print("`preds` has not been deleted!")            
            



    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)
            print(print_str[:-5])

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                lr_list = self.lr_scheduler.get_lr()
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.module_name, self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)