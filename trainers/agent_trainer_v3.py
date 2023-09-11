# this trainer is used to perform option-critic, the first policy is used to determine the number of burst
# the second policy is used to determine the pixel shift trajectory.
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
from trainers.base_agent_trainer_v3 import BaseAgentTrainer
from admin.stats import AverageMeter, StatValue
from admin.tensorboard import TensorboardWriter
import torch
import torch.nn as nn
from torch.distributions import Categorical
import time
import numpy as np
from models_dbsr.loss.image_quality_v2 import PSNR, PixelWiseError, SSIM
import data.camera_pipeline as rgb2raw
import data.synthetic_burst_generation as syn_burst_generation
from data.postprocessing_functions import SimplePostProcess
import cv2
import pickle

class AgentTrainer(BaseAgentTrainer):
    def __init__(self, actors, loaders, high_level_optimizer, option_optimizer, settings, 
                 init_permutation=None, discount_factor=0.99, sr_net=None, 
                 high_level_lr_scheduler=None, option_lr_scheduler=None, iterations=15, 
                 interpolation_type='bilinear', reward_type='psnr', save_results=False, saving_dir=None, penalty_alpha=0.5, high_only=False, low_only=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actors, loaders, high_level_optimizer, option_optimizer, settings, high_level_lr_scheduler, option_lr_scheduler)

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
        
        self.save_results = save_results
        
        self.final_permutations = []
        
        self.initial_psnr_sum = 0
        self.final_psnr_sum = 0
        
        self.saving_dir = saving_dir
        
        self.initial_permutations = [torch.tensor(np.array([[0,0],[0,2]])).repeat(settings.batch_size, 1, 1),
                                     torch.tensor(np.array([[0,0],[0,2],[2,2]])).repeat(settings.batch_size, 1, 1),
                                     torch.tensor(np.array([[0,0],[0,2],[2,2],[2,0]])).repeat(settings.batch_size, 1, 1),
                                     torch.tensor(np.array([[0,0],[0,2],[2,2],[2,0],[1,1]])).repeat(settings.batch_size, 1, 1),
                                     torch.tensor(np.array([[0,0],[0,2],[2,2],[2,0],[1,1],[0,1]])).repeat(settings.batch_size, 1, 1),
                                     torch.tensor(np.array([[0,0],[0,2],[2,2],[2,0],[1,1],[0,1],[1,0]])).repeat(settings.batch_size, 1, 1),
                                     torch.tensor(np.array([[0,0],[0,2],[2,2],[2,0],[1,1],[0,1],[1,0],[1,2]])).repeat(settings.batch_size, 1, 1),
                                     torch.tensor(np.array([[0,0],[0,2],[2,2],[2,0],[1,1],[0,1],[1,0],[1,2],[2,1]])).repeat(settings.batch_size, 1, 1),
                                     torch.tensor(np.array([[0,0],[0,2],[2,2],[2,0],[1,1],[0,1],[1,0],[1,2],[2,1],[0,3]])).repeat(settings.batch_size, 1, 1),
                                     torch.tensor(np.array([[0,0],[0,2],[2,2],[2,0],[1,1],[0,1],[1,0],[1,2],[2,1],[0,3],[3,0]])).repeat(settings.batch_size, 1, 1),
                                     torch.tensor(np.array([[0,0],[0,2],[2,2],[2,0],[1,1],[0,1],[1,0],[1,2],[2,1],[0,3],[3,0],[1,3]])).repeat(settings.batch_size, 1, 1),
                                     torch.tensor(np.array([[0,0],[0,2],[2,2],[2,0],[1,1],[0,1],[1,0],[1,2],[2,1],[0,3],[3,0],[1,3],[3,1]])).repeat(settings.batch_size, 1, 1),
                                     torch.tensor(np.array([[0,0],[0,2],[2,2],[2,0],[1,1],[0,1],[1,0],[1,2],[2,1],[0,3],[3,0],[1,3],[3,1],[2,3]])).repeat(settings.batch_size, 1, 1),
                                     torch.tensor(np.array([[0,0],[0,2],[2,2],[2,0],[1,1],[0,1],[1,0],[1,2],[2,1],[0,3],[3,0],[1,3],[3,1],[2,3],[3,2]])).repeat(settings.batch_size, 1, 1),
                                     torch.tensor(np.array([[0,0],[0,2],[2,2],[2,0],[1,1],[0,1],[1,0],[1,2],[2,1],[0,3],[3,0],[1,3],[3,1],[2,3],[3,2],[3,3]])).repeat(settings.batch_size, 1, 1),]
        
        self.high_only = high_only
        self.low_only = low_only
        
        self.penalty_alpha = penalty_alpha
        
    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)
    
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
        device = actions_batch.device
        movements = {
            0: (-1, 0),  # Left
            1: (1, 0),   # Right
            2: (0, -1),  # Up
            3: (0, 1),    # Down
            4: (0, 0)     # Stay still
        }

        updated_permutations_list = []
        updated_actions_list = []
        for actions, initial_permutations in zip(actions_batch, initial_permutations_batch):
            actions = actions.cpu()
            initial_permutations = initial_permutations.cpu()
            new_actions = actions.clone()
            new_permutations = initial_permutations.clone()
            # updated_permutations = list(initial_permutations)
            # updated_actions = list(actions)
            # print("initial permutations: ", new_permutations)
            for idx, action in enumerate(actions):
                movement = movements[action.item()]
                # print("%sth burst frame movement: " % idx, movement)
                new_permutations[idx+1][0] = initial_permutations[idx+1][0].item() + movement[0]
                new_permutations[idx+1][1] = initial_permutations[idx+1][1].item() + movement[1] 
                # print("new_permutations[idx+1]", new_permutations[idx+1])

                # Clip to boundaries
                new_permutations[idx+1][0] = min(max(new_permutations[idx+1][0].item(), 0), 3)
                new_permutations[idx+1][1] = min(max(new_permutations[idx+1][1].item(), 0), 3)

                # Check for duplicates, if there's a duplicate, select "stay still" action
                duplicated = False
                for i in range(idx+1):
                    if (new_permutations[idx+1][0] == initial_permutations[i][0]) \
                        and (new_permutations[idx+1][1] == initial_permutations[i][1]):
                        duplicated = True
                if duplicated:
                    # print("new_permutations[idx+1]", new_permutations[idx+1])
                    new_permutations[idx+1] = initial_permutations[idx+1].clone()
                    new_actions[idx] = 4  # Stay still action
                else:
                    initial_permutations[idx+1] = new_permutations[idx+1].clone()
            # print("new permutations: ", new_permutations)
            updated_permutations_list.append(new_permutations)
            updated_actions_list.append(new_actions)

        # Convert lists of lists to numpy arrays and then to tensors
        # print("updated_permutations_list", updated_permutations_list)
        updated_permutations_tensor = torch.stack(updated_permutations_list)
        # print("updated_actions_list", updated_actions_list)
        updated_actions_tensor = torch.stack(updated_actions_list)
        # print("type specified_translation: ", new_permutations)

        return updated_permutations_tensor, updated_actions_tensor.to(device)

    def step_environment(self, dists, HR_batch, permutations):
        actions = torch.stack([dist.sample() for dist in dists], dim=1)
        # print("actions: ", actions)
        permutations, actions  = self.update_permutations_and_actions(actions, permutations)
        
        next_state = self.apply_actions_to_env(HR_batch, permutations)
        return next_state, actions, permutations
    
    def apply_actions_to_env(self, HR_batch, permutations_batch):
        """Apply actions to a batch of images."""
        device = HR_batch.device
        # images = images.cpu()
        batch_size = HR_batch.size(0)
        burst_size = permutations_batch.size(1)
        # print("permutations_batch.size(1): ", permutations_batch.size())
        transformed_images = []
        for i in range(batch_size):
            HR = HR_batch[i].clone().cpu()
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

    def _calculate_reward(self, frame_gt, pred_current, pred_last, reward_func=None, batch=True):
        """Calculate the reward as the difference of PSNR between current and last prediction."""
        assert reward_func is not None and reward_func != 'ssim', "You must specify psnr."
        if self.reward_type == 'psnr':
            metric_current = reward_func[self.reward_type](pred_current, frame_gt, batch=batch)
            metric_last = reward_func[self.reward_type](pred_last, frame_gt, batch=batch)
            reward_difference = [curr - last for curr, last in zip(metric_current, metric_last)]
            reward_tensor = torch.stack(reward_difference).unsqueeze(1).to(self.device)
        elif self.reward_type == 'ssim':
            metric_current = reward_func[self.reward_type](pred_current, frame_gt)
            metric_last = reward_func[self.reward_type](pred_last, frame_gt)
            reward = 10*(metric_current - metric_last)    

        return reward_tensor # list(Tensor)

    def compute_returns(self, next_value, rewards, masks=None, gamma=0.99):
        R = next_value
        returns = []
        # print("gamma type: ", type(gamma))
        # print("reward type: ", type(rewards[0]))
        # print("next_value type: ", type(next_value))
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R                       

            returns.insert(0, R)
        return returns

    def save_img_and_metrics(self, initial_pred, final_pred, initial_psnr, final_psnr, meta_info, burst_rgb, gt, final_shifts, name):
        
        self.final_permutations.append(final_shifts.cpu().numpy())
        
        saving_dir = self.saving_dir
        os.makedirs(saving_dir, exist_ok=True) 
        
        process_fn = SimplePostProcess(return_np=True)
        # print("gt: ", gt.size())
        # print("burst_rgb[0]: ", burst_rgb[0].size())
        # print("initial_pred: ", initial_pred.size())
        HR_image = process_fn.process(gt.squeeze(0).cpu(), meta_info)
        LR_image = process_fn.process(burst_rgb[0][0].cpu(), meta_info)
        SR_initial_image = process_fn.process(initial_pred.squeeze(0).cpu(), meta_info)
        SR_final_image = process_fn.process(final_pred.squeeze(0).cpu(), meta_info)
        # name = int(len(os.listdir(saving_dir))/3)
        cv2.imwrite('{}/{}_HR.png'.format(saving_dir, name), HR_image)
        cv2.imwrite('{}/{}_LR.png'.format(saving_dir, name), LR_image)
        cv2.imwrite('{}/{}_SR_initial.png'.format(saving_dir, name), SR_initial_image)
        cv2.imwrite('{}/{}_SR_final.png'.format(saving_dir, name), SR_final_image)
        self.initial_psnr_sum += initial_psnr
        self.final_psnr_sum += final_psnr
  
    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        for idx, _ in enumerate(self.actors):
            self.actors[idx].train(loader.training)
        
        torch.set_grad_enabled(loader.training)
        # torch.autograd.set_detect_anomaly(True)

        self._init_timing()

        # discount_factor = self.discount_factor # set your discount factor

        for i, data in enumerate(loader, 1):

            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings


            if self.save_results:
                initial_pred_meta_info = data['meta_info']
                burst_rgb = data['burst_rgb'].clone()

            batch_size             = data['frame_gt'].size(0)
            self.batch_size        = batch_size
            state                  = data['burst'].clone() 
            log_probs1, log_probs2 = [], []
            values1, values        = [], []
            log_probs              = []
            rewards                = []
            preds                  = []
            penalties              = []
            entropy1_final, entropy2_final = [], []
            
            if self.reward_type == 'psnr':
                reward_func = {'psnr': PSNR(boundary_ignore=40)}
            elif self.reward_type == 'ssim':
                reward_func = {'ssim': SSIM(boundary_ignore=40)}
            else:
                assert 0 == 1, "wrong reward type"
            for it in range(self.iterations):
                option_done = False
                # option 1: determine how many extra burst needed
                dists1, value1 = self.actors[0](state)
                if isinstance(value1, list):
                    value1 = torch.stack(value1)
                if isinstance(dists1, list):
                    actions1_list = []
                    for batch_idx, dist1 in enumerate(dists1):                        
                        actions1_list.append(dist1.sample())                        

                    actions1_list = [action1.to(self.device) for action1 in actions1_list]
                    actions1 = torch.tensor(actions1_list).to(self.device)
                else:
                    actions1 = dists1.sample()
                log_prob1 = torch.zeros(batch_size).to(self.device)
                entropy1 = torch.zeros(batch_size).to(self.device)

                if isinstance(dists1, list):
                    # print("dists1[list]: ", dists1)
                    # 如果 dists1 是一个列表，我们需要手动遍历每一个批次和对应的动作
                    for idx, dist in enumerate(dists1):
                        log_prob1[idx] = dist.log_prob(actions1[idx, 0])
                        entropy1[idx] = dist.entropy()
                    
                    # print("log_prob1[list]: ", log_prob1)
                    # print("entropy1[list]: ", entropy1)
                    # entropy1 = torch.stack(entropy1)
                else:
                    # print("actions1[tensor]: ", actions1)
                    # print("dists1[tensor]: ", dists1.probs.size())
                    # 如果 dists1 是一个 Categorical 实例，我们可以直接使用其方法
                    log_prob1 = dists1.log_prob(actions1[0])
                    entropy1 = dists1.entropy().mean()
                permutations = self.initial_permutations[actions1.item()]
                if self.high_only:
                    with torch.no_grad():
                        pred, _ = self.sr_net(state.clone())
                    preds.append(pred.clone()) 
                state = self.apply_actions_to_env(data['frame_gt'], permutations.clone())
                if self.high_only:
                    with torch.no_grad():
                        pred, _ = self.sr_net(state.clone())
                    preds.append(pred.clone()) 
                    
                in_option_pred_init = None
                
                steps_in_option = 0

                while not option_done and not self.high_only:
                    # print("what is actions1: ", actions1.item())
                    # print("what is state: ", state.size())
                    
                    dists, value = self.actors[actions1.item()+1](state)
                    next_state, actions, permutations = self.step_environment(dists, data['frame_gt'].clone(), permutations.clone())
                    
                    with torch.no_grad():
                        pred, _ = self.sr_net(next_state.clone())
                    preds.append(pred.clone())
                    if steps_in_option == 0:
                        steps_in_option += 1
                        state = next_state.clone()
                        in_option_pred_init = pred.clone()
                        continue
                    reward = self._calculate_reward(data['frame_gt'], preds[-1], preds[-2], reward_func=reward_func, batch=True)
                    log_prob = dists[0].log_prob(actions[:, 0])
                    for id in range(1,len(dists)):
                        log_prob += dists[id].log_prob(actions[:, id])
                    entropy = dists[0].entropy().mean()
                    for idx, dist in enumerate(dists):
                        if idx == 0:
                            continue
                        entropy += dist.entropy().mean()
                    entropy = entropy / (len(dists) - 1)
                    
                    state = next_state.clone()
                    steps_in_option += 1
                    PSNR_current = reward_func[self.reward_type](preds[-1].clone(), data['frame_gt'].clone())
                    PSNR_initial = reward_func[self.reward_type](in_option_pred_init.clone(), data['frame_gt'].clone())
                    if steps_in_option >= 5 or PSNR_current >= 1.1 * PSNR_initial:
                        option_done = True

                    _, next_value = self.actors[actions1.item()+1](state)
                    advantage = reward + self.discount_factor * next_value - value
                    
                    policy_loss = -log_prob * advantage.detach()
                    value_loss = advantage.pow(2)
                    loss = policy_loss + 0.2 * value_loss # - 0.001 * entropy
                    if loader.training:
                        self.option_optimizer.zero_grad()
                        (loss).backward()
                        self.option_optimizer.step()
                    # update statistics
                    batch_size = self.settings.batch_size
                    self._update_stats({'Loss/low_level_loss': loss.item(), 'Loss/policy_loss': policy_loss.item(), 'Loss/value_loss': value_loss.item(),
                                        'Loss/entropy': entropy.item()}, batch_size, loader)

                    # print statistics
                    self._print_stats(i, loader, batch_size)
                
                if self.high_only:
                    if len(preds) == 1:
                        continue
                    else:
                        metric_initial = reward_func[self.reward_type](preds[-2].clone(), data['frame_gt'].clone())
                        metric_final = reward_func[self.reward_type](preds[-1].clone(), data['frame_gt'].clone())
                else:
                    # calculate metric for the initial and final burst
                    metric_initial = reward_func[self.reward_type](in_option_pred_init.clone(), data['frame_gt'].clone())
                    metric_final = reward_func[self.reward_type](preds[-1].clone(), data['frame_gt'].clone())
                
                # 计算每个option的累积奖励（或其他性能指标）
                cumulative_reward_for_option = metric_final - metric_initial  # 这是一个简化的例子

                # 计算高层actor的优势函数
                _, next_value1 = self.actors[0](state)
                if isinstance(next_value1, list):
                    next_value1 = torch.stack(next_value1)
                advantage1 = cumulative_reward_for_option + self.discount_factor * next_value1 - value1.detach()

                # 计算高层actor的损失函数
                if isinstance(dists1, list):
                    expected_burst_list = [(torch.arange(1, 16).float().to(self.device) * dist1.probs) for dist1 in dists1]
                else:
                    expected_burst_list = [(torch.arange(1, 16).float().to(self.device) * prob) for prob in dists1.probs]
                expected_burst = sum(expected_burst_list) / len(expected_burst_list)
                    
                # 计算当前时间步的惩罚项
                penalty = expected_burst.mean()
                high_level_policy_loss = -log_prob1 * advantage1
                high_level_value_loss = advantage1.pow(2)
                high_level_loss = high_level_policy_loss + 0.2 * high_level_value_loss + self.penalty_alpha * penalty# - 0.001 * entropy1 

                # 更新高层actor
                if loader.training:
                    self.high_level_optimizer.zero_grad()
                    high_level_loss.backward()
                    self.high_level_optimizer.step()




                # update statistics
                batch_size = self.settings.batch_size
                self._update_stats({'Loss/high_level_loss': high_level_loss.item(), 'Loss/high_level_policy_loss': high_level_policy_loss.item(), 'Loss/entropy1': entropy1.item(), 'Loss/penalty': penalty.item(), 
                                    ('%s/initial' % self.reward_type): metric_initial.item(), 
                                    ('%s/final' % self.reward_type): metric_final.item(), "Improvement": ((metric_final.item()-metric_initial.item()))}, batch_size, loader)

                # print statistics
                self._print_stats(i, loader, batch_size, high_level=True)
        
            if not loader.training:
                if self.save_results:
                    # save vis results
                    self.save_img_and_metrics(initial_pred=preds[0].clone(), final_pred=preds[-1].clone(), \
                        initial_psnr=metric_initial.item(), final_psnr=metric_final.item(), meta_info=initial_pred_meta_info, \
                            burst_rgb=burst_rgb, gt=data['frame_gt'].clone(), final_shifts=permutations.clone(), name=str(i))
                    # save trajectories
                    f=open("%s/traj.pkl" % self.saving_dir, 'wb')
                    pickle.dump(self.final_permutations, f)
                    f.close()
                    # save metrics
                    f=open("%s/metrics.txt" % self.saving_dir, 'a')
                    print("%sth psnr intial: %s, final: %s, improvement: %s | Average psnr initial: %s, final: %s, improvement: %s" % \
                            (i, metric_initial.item(), metric_final.item(), (metric_final.item()-metric_initial.item()), \
                                self.initial_psnr_sum/float(i), self.final_psnr_sum/float(i), \
                                    self.final_psnr_sum/float(i) - self.initial_psnr_sum/float(i)), file=f)
                    f.close()
            



    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                self.cycle_dataset(loader)
                if self.save_results:
                    assert 1==2, "This means you choose the only eval mode to provide visualization results, so we just run one loader then stop here~"

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

    def _print_stats(self, i, loader, batch_size, high_level=False):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        tag = 'high_level' if high_level else 'low_level'
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s(%s): %d, %d / %d] ' % (loader.name, tag, self.epoch, i, loader.__len__())
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
        
