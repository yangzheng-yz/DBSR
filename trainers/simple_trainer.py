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
from models.loss.image_quality_v2 import PSNR, PixelWiseError
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
    def __init__(self, actor, loaders, optimizer, settings, permutation=None, discount_factor=0.99, sr_net=None, lr_scheduler=None, iterations=7, interpolation_type='bilinear'):
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
        
        self.interpolation_type = interpolation_type
        
        self.discount_factor = discount_factor
        
        self.permutation = permutation

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

    def _update_permutations(self, permutations, actions):
        """Update permutations based on the actions."""
        batch_size, num_images, _ = permutations.shape
        action_offsets = torch.tensor([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]]).to(actions.device)
        for i in range(1, num_images):  # start from 1 because the base frame does not move
            permutations[:, i] = permutations[:, i] + action_offsets[actions[:, i-1]]
        return permutations

    def _calculate_reward(self, frame_gt, pred_current, pred_last, reward_func=None):
        """Calculate the reward as the difference of PSNR between current and last prediction."""
        assert reward_func is not None, "You must specify a reward function."
        psnr_current = reward_func['psnr'](pred_current.clone().detach(), frame_gt)
        psnr_last = reward_func['psnr'](pred_last.clone().detach(), frame_gt)
        reward = psnr_current - psnr_last
        return reward

    def _apply_actions(self, images, permutations, downsample_factor):
        """Apply actions to a batch of images."""
        device = images.device
        # images = images.cpu()
        batch_size = images.size(0)
        burst_size = permutations.size(1)
        transformed_images = []
        for i in range(batch_size):
            image = images[i]
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
        return torch.stack(transformed_images).to(device)


    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        discount_factor = 0.99  # set your discount factor

        for i, data in enumerate(loader, 1):
            # get inputs
            if self.move_data_to_gpu:
                data = {k: v.to(self.device) for k, v in data.items()}

            data['epoch'] = self.epoch
            data['settings'] = self.settings

            batch_size = data['frame_gt'].size(0)

            if self.permutation is not None:
                permutations = torch.tensor(self.permutation).repeat(batch_size, 1, 1).to(self.device)
            if self.permutation is not None:
                permutations = torch.tensor(np.array([[0,0],
                                            [0,2],
                                            [2,2],
                                            [2,0]])).repeat(batch_size, 1, 1).to(self.device)

            rewards = []
            log_probs = []

            preds = []

            pred, _ = self.sr_net(data['burst'])
            preds.append(pred)

            reward_func = {'psnr': PSNR(boundary_ignore=40)}

            for it in range(self.iterations):
                # forward pass, to produce action_pdf
                actions_pdf = self.actor(data)

                # sample and apply actions
                actions = _sample_actions(actions_pdf)
                permutations = _update_permutations(permutations, actions)
                data['burst'] = _apply_actions(data['frame_gt'], permutations, downsample_factor=self.downsample_factor)

                # updates preds and calculate reward
                with torch.no_grad():
                    pred, _ = self.sr_net(data['burst'])
                preds.append(pred)
                reward = _calculate_reward(data['frame_gt'], preds[-1], pred[-2], reward_func=reward_func)
                rewards.append(reward)

                # calculate log probabilities of the sampled actions
                log_prob = torch.log(actions_pdf.gather(2, actions.unsqueeze(-1)).squeeze(-1))
                log_probs.append(log_prob)

            # calculate discounted reward
            reward_iter = sum((discount_factor ** i) * reward for i, reward in enumerate(rewards))
            reward_normalized = reward_iter / float(self.iterations)

            # calculate loss
            log_probs = torch.stack(log_probs)
            loss_iter = -(log_probs * reward_normalized).mean()

            # calculate PSNR for the initial and final burst
            psnr_initial = reward_func['psnr'](preds[0], data['frame_gt'])
            psnr_final = reward_func['psnr'](preds[-1], data['frame_gt'])

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                loss_iter.backward()
                self.optimizer.step()

                # update statistics
                batch_size = self.settings.batch_size
                self._update_stats({'Loss/total': loss_iter.item(), 'PSNR/initial': psnr_initial, 'PSNR/final': psnr_final}, batch_size, loader)

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