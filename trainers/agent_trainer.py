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

class ActorCritic(nn.Module):
    def __init__(self, num_frames, num_channels, hidden_size):
        super(ActorCritic, self).__init__()
        
        # Actor Network
        self.actor_conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.actor_lstm = nn.LSTM(64, hidden_size, batch_first=True)
        self.actor_linear = nn.Linear(hidden_size, 5 * (num_frames - 1))
        
        # Critic Network
        self.critic_conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.critic_linear = nn.Linear(64, 1)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)  # Reshape to [batch_size*num_frames, channels, height, width]
        
        # Actor
        x_actor = self.actor_conv(x)
        x_actor = x_actor.view(batch_size, num_frames, -1)  # Reshape back to [batch_size, num_frames, features]
        _, (h_n, _) = self.actor_lstm(x_actor)
        action_logits = self.actor_linear(h_n.squeeze(0))
        action_logits = action_logits.view(batch_size, num_frames - 1, 5)  # Reshape to [batch_size, num_frames-1, 5]
        probs = F.softmax(action_logits, dim=-1)
        dists = [Categorical(p) for p in probs.split(1, dim=1)]
        
        # Critic
        x_critic = self.critic_conv(x)
        x_critic = x_critic.view(batch_size, num_frames, -1).mean(dim=1)  # Average over frames
        value = self.critic_linear(x_critic)
        
        return dists, value



class AgentTrainer(BaseTrainer):
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
    
    def update_permutations_and_actions(self, actions_batch, initial_permutations_batch):
        movements = {
            0: (-1, 0),  # Left
            1: (1, 0),   # Right
            2: (0, -1),  # Up
            3: (0, 1),    # Down
            4: (0, 0)     # Stay still
        }
        
        updated_permutations_batch = []
        updated_actions_batch = []
        for actions, initial_permutations in zip(actions_batch, initial_permutations_batch):
            updated_permutations = list(initial_permutations)
            updated_actions = list(actions)
            for idx, action in enumerate(actions):
                movement = movements[action]
                updated_permutation = (
                    initial_permutations[idx+1][0] + movement[0],
                    initial_permutations[idx+1][1] + movement[1]
                )
                # Clip to boundaries
                updated_permutation = (
                    min(max(updated_permutation[0], 0), 3),
                    min(max(updated_permutation[1], 0), 3)
                )
                # Check for duplicates, if there's a duplicate, select "stay still" action
                if updated_permutation in updated_permutations:
                    updated_permutation = initial_permutations[idx+1]
                    updated_actions[idx] = 4  # Stay still action
                updated_permutations[idx+1] = updated_permutation
            updated_permutations_batch.append(updated_permutations)
            updated_actions_batch.append(updated_actions)
        
        return updated_permutations_batch, updated_actions_batch

    def step_environment(self, dists, HR_batch, permutations):
        actions = [dist.sample() for dist in dists]
        actions, permutations = update_permutations_and_actions(actions, permutations)
        next_state, reward = self.apply_actions_to_env(HR_batch, permutations)
        return next_state, reward, actions, permutations

    def apply_actions_to_env(self, HR_batch, permutations_batch):
        """Apply actions to a batch of images."""
        device = HR_batch.device
        # images = images.cpu()
        batch_size = HR_batch.size(0)
        burst_size = permutations_batch.size(1)
        transformed_images = []
        for i in range(batch_size):
            HR = HR_batch[i].cpu()
            # print("image type: ", image.device)
            # time.sleep(1000)
            burst_transformation_params = {'max_translation': 24.0,
                                        'max_rotation': 1.0,
                                        'max_shear': 0.0,
                                        'max_scale': 0.0,
                                        'random_pixelshift': False,
                                        'specified_translation': permutations_batch[i]}
            image_burst_rgb, _ = syn_burst_generation.single2lrburstdatabase(HR, burst_size=burst_size,
                                                        downsample_factor=self.downsample_factor,
                                                        transformation_params=burst_transformation_params,
                                                        interpolation_type=self.interpolation_type)
            image_burst = rgb2raw.mosaic(image_burst_rgb.clone())
            transformed_images.append(image_burst)
            transformed_images_stacked = torch.stack(transformed_images).to(device)
        return transformed_images_stacked

 
    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)
        # torch.autograd.set_detect_anomaly(True)

        self._init_timing()

        discount_factor = self.discount_factor # set your discount factor

        for i, data in enumerate(loader, 1):

            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings

            batch_size = data['frame_gt'].size(0)
            log_probs = []
            values    = []
            rewards   = []
            masks     = []
            preds     = []
            entropy   =  0
            pred, _   = self.sr_net(data['burst'])
            preds.append(pred.clone())

            if self.init_permutation is not None:
                permutations = torch.tensor(self.init_permutation).repeat(batch_size, 1, 1)
            else:
                permutations = torch.tensor(np.array([[0,0],
                                            [0,2],
                                            [2,2],
                                            [2,0]])).repeat(batch_size, 1, 1)
            
            if self.reward_type == 'psnr':
                reward_func = {'psnr': PSNR(boundary_ignore=40)}
            elif self.reward_type == 'ssim':
                reward_func = {'ssim': SSIM(boundary_ignore=40)}
            else:
                assert 0 == 1, "wrong reward type"

            state = data['burst'].clone()

            for it in range(self.iterations):
                dists, value = self.actor(state)
                next_state, reward, actions, permutations = self.step_environment(dists, data['frame_gt'], permutations)
                
                # sample and apply actions
                actions = self._sample_actions_v2(actions_pdf)
                selected_indices = self._update_selections(selected_indices, actions)
                print("%s iteration selected indices: " % it, selected_indices)
                batch_indices = torch.arange(batch_size)[:, None]
                selected_data = data['burst'][batch_indices, selected_indices].clone()

                # updates preds and calculate reward
                with torch.no_grad():
                    pred, _ = self.sr_net(selected_data)

                preds.append(pred.clone())

                reward = self._calculate_reward(data['frame_gt'], preds[-1], preds[-2], reward_func=reward_func)

                rewards.append(reward)

                state = next_state
                
                # calculate log probabilities of the sampled actions
                log_prob = torch.log(actions_pdf.gather(2, actions.unsqueeze(-1)).squeeze(-1))
                log_probs.append(log_prob)                

            # calculate discounted reward
            reward_iter = sum((discount_factor ** i) * reward for i, reward in enumerate(rewards))
            reward_normalized = reward_iter / float(self.iterations)

            # calculate loss
            log_probs = torch.stack(log_probs)
            loss_iter = -(log_probs * reward_iter).sum()

            # calculate metric for the initial and final burst
            metric_initial = reward_func[self.reward_type](pred, data['frame_gt'])
            metric_final = reward_func[self.reward_type](preds[-1], data['frame_gt'])
            metrix_init_final = reward_func[self.reward_type](preds[-1], pred)


            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                (loss_iter).backward()
                self.optimizer.step()

            # update statistics
            batch_size = self.settings.batch_size
            self._update_stats({'Loss/total': loss_iter.item(), ('%s/initial' % self.reward_type): metric_initial.item(), ('%s/final' % self.reward_type): metric_final.item(), "Improvement": metric_final.item()-metric_initial.item(), "Init_final": metrix_init_final}, batch_size, loader)

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
        
