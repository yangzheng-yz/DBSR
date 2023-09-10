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
from trainers.base_agent_trainer_v2 import BaseAgentTrainer
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
    def __init__(self, actors, loaders, optimizer, settings, 
                 init_permutation=None, discount_factor=0.99, sr_net=None, 
                 lr_scheduler=None, iterations=15, 
                 interpolation_type='bilinear', reward_type='psnr', save_results=False, saving_dir=None, penalty_alpha=0.5):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actors, loaders, optimizer, settings, lr_scheduler)

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
        
        self.All_possible_shift = torch.tensor([
                                [1,0],
                                [2,0],
                                [3,0],
                                [0,1],
                                [1,1],
                                [2,1],
                                [3,1],
                                [0,2],
                                [1,2],
                                [2,2],
                                [3,2],
                                [0,3],
                                [1,3],
                                [2,3],
                                [3,3],
                            ])
    
        self.alpha = penalty_alpha
        
    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def step_environment(self, dists2, HR_batch, actions1):
        """
        Update the environment state based on actions from option2, and return the new state, actions2, and permutations.
        
        Parameters:
        - dists2: Probability distribution of actions for option2, shape [batch_size, 15]
        - frame_gt: Ground truth frames, shape could vary based on your specific implementation
        - actions1: Actions taken based on option1, shape [batch_size, 1]
        
        Returns:
        - next_state: Updated state
        - actions2: Actions taken based on option2, shape [batch_size, n] where n is determined by actions1
        - permutations: Final permutations or shifts, shape [batch_size, n, 2]
        """
        self.All_possible_shift = self.All_possible_shift.to(self.device)
        zero_row = torch.tensor([[0, 0]], device=self.device)
        try:
            batch_size = len(dists2.probs)
        except:
            batch_size = len(dists2)
        assert batch_size == self.batch_size, "The output dists2[%s] has different batch size compared with input data[%s]." % (batch_size, self.batch_size)
        
        # Initialize lists to hold results
        actions2_list = []
        permutations_list = []
        
        # Iterate through the batch
        for i in range(batch_size):
            # Get the number of bursts for this sample
            num_bursts = actions1[i].item() + 1 # here +1 is for cancel 0
            # print("num_bursts: ", num_bursts)
            # Get the probabilities for this sample from dists2
            try:
                probs = dists2.probs[i]
            except:
                probs = dists2[i].probs
                probs = probs.squeeze(0)
            # Get the top-n action indices based on the probabilities
            # print("probs: ", probs)
            top_n_indices = torch.argsort(probs, descending=True)[:num_bursts]
            
            # Get the corresponding actions and permutations
            selected_actions = top_n_indices
            # print("what is All_possible_shift: ", type(self.All_possible_shift))
            # print("what is top_n_indices: ", type(top_n_indices))
            selected_permutations = self.All_possible_shift[top_n_indices]
            # print("zero_row: %s, selected_permutations: %s" % (zero_row.size(), selected_permutations.size()))
            selected_permutations = torch.cat((zero_row, selected_permutations), dim=0) # here another +1 is for [0,0]
            # print("what is selected_permutations: ", selected_permutations)
            actions2_list.append(selected_actions)
            permutations_list.append(selected_permutations)
            
        # Keep the results as lists; each element can have a different length corresponding to each batch sample's actions1
        actions2 = actions2_list
        permutations = permutations_list
        
        # Here, you would typically update 'next_state' based on 'frame_gt', 'actions1', and 'actions2'.
        next_state = self.apply_actions_to_env(HR_batch, permutations)
        return next_state, actions2, permutations

    def apply_actions_to_env(self, HR_batch, permutations_batch):
        """Apply actions to a batch of images."""
        device = HR_batch.device
        # images = images.cpu()
        batch_size = HR_batch.size(0)
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
            image_burst_rgb, _ = syn_burst_generation.single2lrburstdatabase(HR, burst_size=len(permutations_batch[i]),
                                                        downsample_factor=self.downsample_factor,
                                                        transformation_params=burst_transformation_params,
                                                        interpolation_type=self.interpolation_type)
            image_burst = rgb2raw.mosaic(image_burst_rgb.clone())
            transformed_images.append(image_burst)
        # transformed_images_stacked = torch.stack(transformed_images).to(device)
        
        return transformed_images

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

        discount_factor = self.discount_factor # set your discount factor

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
            values1, values2       = [], []
            rewards                = []
            preds                  = []
            penalties              = []
            entropy1_final, entropy2_final = [], []
            with torch.no_grad():
                pred, _   = self.sr_net(data['burst'])
            preds.append(pred.clone())
            
            if self.reward_type == 'psnr':
                reward_func = {'psnr': PSNR(boundary_ignore=40)}
            elif self.reward_type == 'ssim':
                reward_func = {'ssim': SSIM(boundary_ignore=40)}
            else:
                assert 0 == 1, "wrong reward type"

            # print("device of state: ", state.size())
            for it in range(self.iterations):
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


                # print("what is action1: ", actions1) # is [batch_size, 1]
                
                # 计算当前时间步的期望 burst 数量
                # print("what is dists1[0].probs", dists1.probs)
                # time.sleep(1000)
                if isinstance(dists1, list):
                    expected_burst_list = [(torch.arange(1, 16).float().to(self.device) * dist1.probs) for dist1 in dists1]
                else:
                    expected_burst_list = [(torch.arange(1, 16).float().to(self.device) * prob) for prob in dists1.probs]
                expected_burst = sum(expected_burst_list) / len(expected_burst_list)
                    
                # 计算当前时间步的惩罚项
                penalty = expected_burst.mean()
                
                penalties.append(penalty)

                # option 2: determine the pixel shift

                # print("state size: ", state.size())
                # print("action1 size: ", actions1.size()) # should be [batch_size, 1]
                actions1 = actions1.unsqueeze(1)
                # print("what is dists1[0].probs", actions1.size())
                dists2, value2 = self.actors[1](state, actions1)
                if isinstance(value2, list):
                    value2 = torch.stack(value2)
                next_state, actions2, permutations = self.step_environment(dists2, data['frame_gt'].clone(), actions1)
                # updates preds and calculate reward
                pred_list = []
                with torch.no_grad():
                    # print("device of model: ", next(self.sr_net.parameters()).device)
                    if isinstance(next_state, list):
                        for single_next_state in next_state:
                            single_next_state = single_next_state.to(self.device)
                            # print("device of single_next_state: ", single_next_state.size())
                            single_pred, _ = self.sr_net(single_next_state.unsqueeze(0))
                            pred_list.append(single_pred)
                        pred = torch.stack(pred_list).to(self.device)
                    else:
                        next_state = next_state.to(self.device)
                        pred = self.sr_net(next_state.clone())
                preds.append(pred.clone())

                reward = self._calculate_reward(data['frame_gt'], preds[-1], preds[-2], reward_func=reward_func, batch=True)
                # print("what is actions: ", actions.device)
                # print("what is dist: ", dists[0].device)
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
                    log_prob1 = dists1.log_prob(actions1[:,0])
                    entropy1 = dists1.entropy()
                    # print("log_prob1[tensor]: ", log_prob1)
                    # print("entropy1[tensor]: ", entropy1)
                entropy1_final.append(entropy1.mean())
                # print("what is actions2: ", actions2)
                # print("what is dists2: ", dists2)
                
                log_prob_sums = []

                # 遍历每个批次
                for batch_idx, action_tensor in enumerate(actions2):
                    if isinstance(dists2, list):
                        dist_for_this_batch = dists2[batch_idx]
                    else:
                        # 从整体分布中取出这个批次对应的分布
                        dist_for_this_batch = Categorical(probs=dists2.probs[batch_idx])
                    
                    # 计算这个批次中每个动作的对数概率
                    log_probs = dist_for_this_batch.log_prob(action_tensor)
                    # print("what is log_probs: ", log_probs)
                    # 计算对数概率的和
                    log_prob_sum = log_probs.sum().item()
                    
                    log_prob_sums.append(log_prob_sum)

                log_prob2 = torch.tensor(log_prob_sums).to(self.device)  # 转换为 PyTorch 张量
                # print("what is log_prob2: ", log_prob2)
                
                if isinstance(dists2, list):
                    entropy2_list = []
                    for batch_idx, _ in enumerate(dists2):
                        entropy2_list.append(dists2[batch_idx].entropy())
                    # print("what is entropy2_list: ", entropy2_list)
                    entropy2 = torch.stack(entropy2_list) 
                else:
                    entropy2 = dists2.entropy()  
                entropy2_final.append(entropy2.mean())

                # print("what is entropy2: ", entropy2)  

                rewards.append(reward) # (timestep, batch_size, 1)
                log_probs1.append(log_prob1) # (timestep, batch, )
                log_probs2.append(log_prob2) # (timestep, batch, )
                values1.append(value1)
                values2.append(value2)
                
                state = [tensor.clone().to(self.device) for tensor in next_state]
            
            next_dists1, next_value1 = self.actors[0](state)
            if isinstance(next_dists1, list):
                next_actions1_list = []
                for batch_idx, dist1 in enumerate(next_dists1):                        
                    next_actions1_list.append(dist1.sample())                        

                next_actions1_list = [action1.to(self.device) for action1 in next_actions1_list]
                next_actions1 = torch.tensor(next_actions1_list).to(self.device)
            else:
                next_actions1 = next_dists1.sample()
            next_dists2, next_value2 = self.actors[1](state, next_actions1)
            # print("what is the next_value1 before squeeze: ", next_value1)
            if isinstance(next_value1, list):
                next_value1 = torch.stack(next_value1)
                # next_value1 = next_value1.squeeze(0)
            if isinstance(next_value2, list):
                next_value2 = torch.stack(next_value2)
                # next_value2 = next_value2.squeeze(0)
            # print("what is the next_value1: ", next_value1)
            # print("what is the rewards: ", type(rewards))
            # print("what is the discount_factor: ", type(discount_factor))
            returns1 = self.compute_returns(next_value1, rewards, gamma=discount_factor)
            returns2 = self.compute_returns(next_value2, rewards, gamma=discount_factor)

            # print("returns info, size %s, type %s" % (len(returns), type(returns)))
            # print("log_probs info, size %s, type %s" % (len(log_probs), type(log_probs)))
            # print("values info, size %s, type %s" % (len(values), type(values)))

            # print("what is log_probs1 before cat: ", log_probs1)
            # print("what is log_probs2 before cat: ", log_probs2)
            # print("what is returns1 before cat: ", returns1)
            # print("what is returns2 before cat: ", returns2)
            # print("what is values1 before cat: ", values1)
            # print("what is values2 before cat: ", values2)
            log_probs1 = torch.cat(log_probs1)
            log_probs2 = torch.cat(log_probs2)
            returns1   = torch.cat(returns1).detach()
            returns2   = torch.cat(returns2).detach()
            values1    = torch.cat(values1)
            values2    = torch.cat(values2)
            # print("what is log_probs1 before cat: ", log_probs1)
            # print("what is log_probs2 before cat: ", log_probs2)
            # print("what is returns1 before cat: ", returns1)
            # print("what is returns2 before cat: ", returns2)
            # print("what is values1 before cat: ", values1)
            # print("what is values2 before cat: ", values2)
            
            advantage1 = returns1 - values1
            advantage2 = returns2 - values2

            actor_loss1  = -(log_probs1 * advantage1.detach()).mean()
            critic_loss1 = advantage1.pow(2).mean()
            actor_loss2  = -(log_probs2 * advantage2.detach()).mean()
            critic_loss2 = advantage2.pow(2).mean()

            # 计算总惩罚
            total_penalty = torch.stack(penalties).mean()

            entropy1_final = torch.stack(entropy1_final)
            entropy2_final = torch.stack(entropy2_final)

            loss = actor_loss1 + 0.2 * actor_loss2 + 0.1 * critic_loss1 + 0.1 * critic_loss2 - 0.001 * entropy1_final.mean() - 0.001 * entropy2_final.mean() + self.alpha * total_penalty
            # print("what is actor_loss1: ", actor_loss1)
            # print("what is actor_loss2: ", actor_loss2)
            # print("what is critic_loss1: ", critic_loss1)
            # print("what is critic_loss2: ", critic_loss2)
            # print("what is entropy1: ", entropy1)
            # print("what is entropy2: ", entropy2)
            # print("what is entropy2: ", entropy2)
            # print("what is loss: ", loss)
            # calculate metric for the initial and final burst
            metric_initial = reward_func[self.reward_type](preds[0].clone(), data['frame_gt'].clone())
            metric_final = reward_func[self.reward_type](preds[-1].clone(), data['frame_gt'].clone())

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                (loss).backward()
                self.optimizer.step()

            # update statistics
            batch_size = self.settings.batch_size
            self._update_stats({'Loss/total': loss.item(), 'Loss/actor1': actor_loss1.item(), 'Loss/critic1': critic_loss1.item(), 'Loss/entropy1': entropy1_final.mean().item(),
                                'Loss/actor2': actor_loss2.item(), 'Loss/critic2': critic_loss2.item(), 'Loss/entropy2': entropy2_final.mean().item(), ('%s/initial' % self.reward_type): metric_initial.item(), 
                                ('%s/final' % self.reward_type): metric_final.item(), "Improvement": ((metric_final.item()-metric_initial.item()))}, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)
        
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
        
