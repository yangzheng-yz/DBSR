import os
from collections import OrderedDict
from trainers.base_agent_trainer_v3 import BaseAgentTrainer
from admin.stats import AverageMeter, StatValue
from admin.tensorboard import TensorboardWriter
import torch
import torch.nn as nn
import time
import numpy as np
from models_dbsr.loss.image_quality_v2 import PSNR, PixelWiseError, SSIM
import data.camera_pipeline as rgb2raw
import data.synthetic_burst_generation as syn_burst_generation
from data.postprocessing_functions import SimplePostProcess
import cv2
import pickle
from tqdm import tqdm
import torch.nn.functional as F
from actors.dbsr_actors import qValueNetwork
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


class AgentSAC(BaseAgentTrainer):
    def __init__(self, actors, loaders, actor_optimizer, critic_1_optimizer, critic_2_optimizer, log_alpha_optimizer, settings, 
                actor_lr_scheduler=None, critic_1_lr_scheduler=None, critic_2_lr_scheduler=None, log_alpha_lr_scheduler=None, log_alpha=0,
                discount_factor=0.99, sr_net=None, 
                lr_scheduler=None, iterations=15, 
                interpolation_type='bilinear', reward_type='psnr', save_results=False, saving_dir=None, one_step_length=1/4, base_length=1/4,
                target_entropy=-1, tau=0.005, init_permutation=None, minimal_size=50, sample_size=4, accelerator=None,
                loader_attributes=None, actors_attr=None, gpus_num=8,
                inital_epoch=0):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actors, loaders, actor_optimizer, critic_1_optimizer, critic_2_optimizer, \
            log_alpha_optimizer, settings, actor_lr_scheduler, critic_1_lr_scheduler, critic_2_lr_scheduler, log_alpha_lr_scheduler, log_alpha, actors_attr, inital_epoch)

        self._set_default_settings()
        
        self.gpus_num = gpus_num
        
        
        self.loader_attributes = loader_attributes
        self.accelerator = accelerator
        self.sample_size = sample_size
        
        self.target_critic_1 = actors[-2]
        self.target_critic_2 = actors[-1]
        
        # self.target_critic_1.load_state_dict(self.actors[1].state_dict())
        # self.target_critic_2.load_state_dict(self.actors[2].state_dict())
        
        self.target_entropy = target_entropy
        
        # Initialize statistics variables
        self.stats = OrderedDict({self.loader_attributes[idx]['name']: None for idx, _ in enumerate(self.loaders)})

        # Initialize tensorboard
        tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
        self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [self.loader_attributes[idx]['name'] for idx, l in enumerate(loaders)])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        
        self.iterations = iterations
        
        self.downsample_factor = settings.downsample_factor
        
        assert sr_net is not None, "You must specify a pretrained SR model to calculate reward"
        self.sr_net = sr_net
        # self.sr_net = self.sr_net.to(self.device)
        
        self.interpolation_type = interpolation_type
        
        self.discount_factor = discount_factor
        
        self.init_permutation = init_permutation
        
        self.reward_type = reward_type
        
        self.save_results = save_results
        
        self.final_permutations = []
        self.final_permutations_int_length = []
        self.one_step_length = one_step_length
        
        self.initial_psnr_sum = 0
        self.final_psnr_sum = 0
        
        self.saving_dir = saving_dir
        self.base_length = base_length
        self.one_step_length_in_grid = float(one_step_length / base_length)
        # self.base_length = base_length
        self.tau = tau
        
        self.minimal_size = minimal_size
        tb_save_root_dir = os.path.join(settings.env.workspace_dir, 'tensorboard')
        if not os.path.exists(tb_save_root_dir):
            os.makedirs(tb_save_root_dir, exist_ok=True)
        tb_save_dir = os.path.join(tb_save_root_dir, settings.project_path)
        os.makedirs(tb_save_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=tb_save_dir)
        
        
    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                'print_stats': None,
                'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)
    
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
                new_permutations[idx+1][0] = initial_permutations[idx+1][0].item() + movement[0] * self.one_step_length_in_grid
                new_permutations[idx+1][1] = initial_permutations[idx+1][1].item() + movement[1] * self.one_step_length_in_grid
                # print("new_permutations[idx+1]", new_permutations[idx+1])

                # Clip to boundaries
                new_permutations[idx+1][0] = min(max(new_permutations[idx+1][0].item(), 0.), 4 - self.one_step_length_in_grid)
                new_permutations[idx+1][1] = min(max(new_permutations[idx+1][1].item(), 0.), 4 - self.one_step_length_in_grid)

                # Check for duplicates, if there's a duplicate, select "stay still" action
                duplicated = False
                for i in range(idx+1):
                    if (new_permutations[idx+1][0] == initial_permutations[i][0]) \
                        and (new_permutations[idx+1][1] == initial_permutations[i][1]):
                        duplicated = True
                        break
                if not duplicated:
                    for i in range(idx+2, len(initial_permutations)):
                        if (new_permutations[idx+1][0] == initial_permutations[i][0]) \
                            and (new_permutations[idx+1][1] == initial_permutations[i][1]):
                            duplicated = True
                            break
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
    
    # def update_permutations_and_actions(self, actions_batch, initial_permutations_batch):
    #     device = actions_batch.device
    #     movements = {
    #         0: (-1, 0),  # Left
    #         1: (1, 0),   # Right
    #         2: (0, -1),  # Up
    #         3: (0, 1),    # Down
    #         4: (0, 0)     # Stay still
    #     }

    #     updated_permutations_list = []
    #     updated_actions_list = []
    #     for actions, initial_permutations in zip(actions_batch, initial_permutations_batch):
    #         actions = actions.cpu()
    #         initial_permutations = initial_permutations.cpu()
    #         new_actions = actions.clone()
    #         new_permutations = initial_permutations.clone()
    #         # updated_permutations = list(initial_permutations)
    #         # updated_actions = list(actions)
    #         # print("initial permutations: ", new_permutations)
    #         for idx, action in enumerate(actions):
    #             movement = movements[action.item()]
    #             # print("%sth burst frame movement: " % idx, movement)
    #             # new_permutations[idx+1][0] = initial_permutations[idx+1][0].item() + movement[0] * self.one_step_length_in_grid
    #             # new_permutations[idx+1][1] = initial_permutations[idx+1][1].item() + movement[1] * self.one_step_length_in_grid
    #             new_x = initial_permutations[idx+1][0] + movement[0] * self.one_step_length_in_grid
    #             new_y = initial_permutations[idx+1][1] + movement[1] * self.one_step_length_in_grid

    #             # print("new_permutations[idx+1]", new_permutations[idx+1])

    #             # Clip to boundaries
    #             # new_permutations[idx+1][0] = min(max(new_permutations[idx+1][0].item(), 0.), 4 - self.one_step_length_in_grid)
    #             # new_permutations[idx+1][1] = min(max(new_permutations[idx+1][1].item(), 0.), 4 - self.one_step_length_in_grid)
    #             # Check for out-of-boundary and adjust the movement if necessary
    #             if new_x < 0 or new_x >= 4 or new_y < 0 or new_y >= 4:
    #                 # If movement goes out of boundary, reset to the initial position
    #                 new_actions[idx] = 4  # Stay still action
    #                 new_permutations[idx+1] = initial_permutations[idx+1].clone()
                
    #             duplicated = False
    #             for i in range(idx+1):
    #                 if (new_permutations[idx+1][0] == initial_permutations[i][0]) \
    #                     and (new_permutations[idx+1][1] == initial_permutations[i][1]):
    #                     duplicated = True
    #                     break  # No need to check further if duplicate is found
                
    #             if duplicated:
    #                 # Check surrounding positions for a valid move
    #                 possible_moves = [0, 1, 2, 3]  # Corresponds to left, right, up, down
    #                 for move in possible_moves:
    #                     potential_move = movements[move]
    #                     new_x = initial_permutations[idx+1][0] + potential_move[0] * self.one_step_length_in_grid
    #                     new_y = initial_permutations[idx+1][1] + potential_move[1] * self.one_step_length_in_grid
                        
    #                     # Check if the new position is within bounds and not occupied
    #                     if 0 <= new_x <= 4 - self.one_step_length_in_grid and \
    #                     0 <= new_y <= 4 - self.one_step_length_in_grid:
    #                         occupied = False
    #                         for perm in initial_permutations:
    #                             if new_x == perm[0] and new_y == perm[1]:
    #                                 occupied = True
    #                                 break
                            
    #                         if not occupied:
    #                             # Update the action to the new move if the position is free
    #                             new_actions[idx] = move
    #                             new_permutations[idx+1] = torch.tensor([new_x, new_y])
    #                             break

    #                 if occupied:
    #                     # If all positions around are occupied, stay still
    #                     new_actions[idx] = 4  # Stay still action
    #                     new_permutations[idx+1] = initial_permutations[idx+1].clone()
    #                     print("Warning it should happen!!!!!!!!!!!!!!!!!!!")
    #             else:
    #                 initial_permutations[idx+1] = new_permutations[idx+1].clone()
    #         # print("new permutations: ", new_permutations)
    #         updated_permutations_list.append(new_permutations)
    #         updated_actions_list.append(new_actions)

    #     # Convert lists of lists to numpy arrays and then to tensors
    #     # print("updated_permutations_list", updated_permutations_list)
    #     updated_permutations_tensor = torch.stack(updated_permutations_list)
    #     # print("updated_actions_list", updated_actions_list)
    #     updated_actions_tensor = torch.stack(updated_actions_list)
    #     print("type specified_translation: ", new_permutations)
    #     # 先将张量转移到CPU，再转换为numpy数组，最后展开并输出
    #     # print(f"Updated permutations: {updated_permutations_tensor.cpu().numpy().reshape(-1)}")

    #     return updated_permutations_tensor, updated_actions_tensor.to(device)

    def step_environment(self, dists, HR_batch, permutations, iter, add_noise=True, meta_info=None, save_results=False):

        if save_results:
            # 如果在测试阶段，选择概率最高的动作
            actions = torch.stack([torch.argmax(dist.probs, dim=-1) for dist in dists], dim=1)
        else:
            # 如果在训练阶段，随机抽样动作
            actions = torch.stack([dist.sample() for dist in dists], dim=1)

        permutations, actions = self.update_permutations_and_actions(actions, permutations)
        
        next_state = self.apply_actions_to_env(HR_batch, permutations, add_noise=add_noise, meta_info=meta_info)
        done = 0
        if iter == self.iterations:
            done = 1

        return next_state, actions, permutations, done


    def apply_actions_to_env(self, HR_batch, permutations_batch, add_noise=True, meta_info=None):
        """Apply actions to a batch of images."""
        device = HR_batch.device
        # images = images.cpu()
        batch_size = HR_batch.size(0)
        # print("permutations_batch: ", permutations_batch)
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
            # Add noise
            if add_noise:
                shot_noise_level = meta_info['shot_noise_level'][i].cpu()
                read_noise_level = meta_info['read_noise_level'][i].cpu()
                image_burst = rgb2raw.add_noise(image_burst, shot_noise_level, read_noise_level)

            # Clip saturated pixels.
            image_burst = image_burst.clamp(0.0, 1.0)
            
            transformed_images.append(image_burst)
        transformed_images_stacked = torch.stack(transformed_images).to(device)
        
        return transformed_images_stacked

    def _calculate_reward(self, frame_gt, pred_current, pred_last, reward_func=None, batch=True, tolerance=0.5):
        """Calculate the reward as the difference of PSNR between current and last prediction."""
        assert reward_func is not None and reward_func != 'ssim', "You must specify psnr."
        if self.reward_type == 'psnr':
            metric_current = reward_func[self.reward_type](pred_current, frame_gt, batch=batch)
            metric_last = reward_func[self.reward_type](pred_last, frame_gt, batch=batch)
            reward_difference = [curr - last + tolerance for curr, last in zip(metric_current, metric_last)]
            reward_tensor = torch.stack(reward_difference).unsqueeze(1).to(frame_gt.device)
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
        
        self.final_permutations_int_length.append((final_shifts.cpu().numpy() * (1 / self.one_step_length_in_grid)).astype(np.int32))
        self.final_permutations.append(final_shifts.cpu().numpy())
        
        saving_dir = self.saving_dir
        os.makedirs(saving_dir, exist_ok=True) 
        for key, value in meta_info.items():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                meta_info[key] = value.cpu()

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

    def generate_objective(self, pre_actor, frame_gt, pre_actor_step=4, pre_init_permutation=None):
        print("You can see this message cause you used the pre_actor function!")
        
        state = self.apply_actions_to_env(frame_gt, pre_init_permutation.clone())
        permutations = pre_init_permutation.clone()
        
        for it in range(pre_actor_step):
            print("What is the permutations: ", permutations.size())
            with torch.no_grad():
                dists, value = pre_actor(state)
            next_state, actions, permutations = self.step_environment(dists, frame_gt.clone(), permutations.clone())
            state = next_state.clone()
        return permutations, state
    
    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                self.cycle_dataset(loader)
                if self.save_results:
                    assert 1==2, "This means you choose the only eval mode to provide visualization results, so we just run one loader then stop here~"

        self._stats_new_epoch()
        self._write_tensorboard()
        
    def train_sac(self, max_epochs, replay_buffer):
        
        self.train_off_policy_agent(self.loaders, max_epochs, replay_buffer, minimal_size=self.minimal_size)

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    # def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
    #     # Initialize stats if not initialized yet
    #     if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
    #         self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

    #     for name, val in new_stats.items():
    #         # print(f"Debug what the name: {name}, value: {val}")
    #         if name not in self.stats[loader.name].keys():
    #             self.stats[loader.name][name] = AverageMeter()
    #         self.stats[loader.name][name].update(val, batch_size)

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader['name'] not in self.stats.keys() or self.stats[loader['name']] is None:
            self.stats[loader['name']] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            # print(f"Debug what the name: {name}, value: {val}")
            if name not in self.stats[loader['name']].keys():
                self.stats[loader['name']][name] = AverageMeter()
            self.stats[loader['name']][name].update(val, batch_size)


    # def _print_stats(self, i, loader, batch_size):
    #     self.num_frames += batch_size
    #     current_time = time.time()
    #     batch_fps = batch_size / (current_time - self.prev_time)
    #     average_fps = self.num_frames / (current_time - self.start_time)
    #     self.prev_time = current_time
    #     if i % self.settings.print_interval == 0 or i == loader.__len__():
    #         print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
    #         print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
    #         for name, val in self.stats[loader.name].items():
    #             if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
    #                 print_str += '%s: %.5f  ,  ' % (name, val.avg)
    #         print(print_str[:-5])

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader['length']:
            print_str = '[%s: %d, %d / %d] ' % (loader['name'], self.epoch, i, loader['length'])
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader['name']].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)
            print(print_str[:-5])

    def _stats_new_epoch(self):
        # Record learning rate
        # for loader in self.loaders:
        #     if loader.training:
        #         lr_list = self.actor_lr_scheduler.get_lr()
        #         for i, lr in enumerate(lr_list):
        #             var_name = 'LearningRate/group{}'.format(i)
        #             if var_name not in self.stats[loader.name].keys():
        #                 self.stats[loader.name][var_name] = StatValue()
        #             self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.module_name, self.settings.script_name, self.settings.description)
        # print(f"Debug what is self.stats: {self.stats}")
        # for key, value in self.stats.items():
        #     print("Key:", key)
        #     if value:
        #         for sub_key, sub_value in value.items():
        #             print("  Sub-key:", sub_key)
        #             print("  Sub-value:", sub_value)
        #     else:
        #         print("  No sub-values")
        self.tensorboard_writer.write_epoch(self.stats, self.epoch)

    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actors[0](next_states)
        # 根据概率直接计算log_probs
        next_log_probs = torch.log(next_probs + 1e-8)

        # Calculate probabilities and log probabilities for each distribution
        # probs = [dist.probs.squeeze(1) for dist in dists]
        # log_probs = [dist.logits.squeeze(1)+1e-8 for dist in dists]  # logits are the log probabilities in the Categorical distribution
        # print(f"Debug probs[0]: {probs[0].size()}")
        # print(f"Debug log_probs: {log_probs[0].size()}")
        # Calculate entropy for each distribution
        entropy = [-torch.sum(p * lp, dim=-1, keepdim=False) for p, lp in zip(next_probs, next_log_probs)]
        entropy = [torch.sum(e, dim=-1, keepdim=True) / (self.settings.burst_sz - 1) for e in entropy]
        entropy = torch.stack(entropy)
        # print(f"Debug entropy: {entropy}")
        # entropy = sum(entropy) # [batch_size, 1]
        # print(f"Debug entropy: {entropy}")        
        # next_probs = torch.stack(next_probs).to(self.accelerator.device)
        
        # next_probs = next_probs.permute(1, 0, 2) # [batch_size, self.burst_sz-1, action_dim]
        
        # next_probs = self.actors[0](next_states)
        # next_log_probs = torch.log(next_probs + 1e-8)
        # entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states) # [batch_size, self.burst_sz-1, action_dim]
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                            dim=-1,
                            keepdim=False)
        min_qvalue, _ = torch.min(min_qvalue, dim=1, keepdim=True) # to avoid overestimate, we should select the minimum agent sum qvalue
        # print(f"Debug min_qvalue: {min_qvalue}")
        # print(f"Debug entropy: {entropy}")
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        
        td_target = rewards + self.discount_factor * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                    net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.stack(transition_dict['states']).to(self.accelerator.device)
        actions = torch.stack(transition_dict['actions']).to(
            self.accelerator.device) 
        rewards = torch.stack(transition_dict['rewards']).to(self.accelerator.device)
        next_states = torch.stack(transition_dict['next_states']).to(self.accelerator.device)
        dones = torch.tensor(transition_dict['dones'],
                            ).view(-1, 1).to(self.accelerator.device)

        # print(f"Debug next_states: {next_states.size()}")
        # print(f"Debug actions: {actions.size()}")
        # print(f"Debug rewards: {rewards.size()}")
        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        # critic_1_q_values = self.actors[1](states).gather(1, actions) # the critic network output all qvalue, then we select according to actions
        critic_1_q_values = torch.gather(self.actors[1](states), \
            dim=2, index=actions).squeeze(-1)
        critic_1_q_values = critic_1_q_values.sum(dim=1, keepdim=True)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        # critic_2_q_values = self.actors[2](states).gather(1, actions)
        critic_2_q_values = torch.gather(self.actors[2](states), \
            dim=2, index=actions).squeeze(-1)
        critic_2_q_values = critic_2_q_values.sum(dim=1, keepdim=True)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        # critic_1_loss.backward()
        self.accelerator.backward(critic_1_loss)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        # critic_2_loss.backward()
        self.accelerator.backward(critic_2_loss)
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actors[0](states)
        # 根据概率直接计算log_probs
        log_probs = torch.log(probs + 1e-8)
        # Calculate probabilities and log probabilities for each distribution
        # probs = [dist.probs.squeeze(1) for dist in dists]
        # log_probs = [dist.logits.squeeze(1)+1e-8 for dist in dists]
        # log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = [-torch.sum(p * lp, dim=-1, keepdim=False) for p, lp in zip(probs, log_probs)]
        entropy = [torch.sum(e, dim=-1, keepdim=True) / (self.settings.burst_sz - 1) for e in entropy]
        entropy = torch.stack(entropy)
        q1_value = self.actors[1](states)
        q2_value = self.actors[2](states)
        # probs = torch.stack(probs)
        # probs = probs.permute(1, 0, 2) # [batch_size, self.burst_sz-1, action_dim]
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                            dim=-1,
                            keepdim=False)  # 直接根据概率计算期望
        min_qvalue, _ = torch.min(min_qvalue, dim=1, keepdim=True)
        
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue) # TODO: why actor loss continue upup!!!!!!!!!!!!!!
        self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        self.accelerator.backward(actor_loss)
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        # alpha_loss.backward()
        self.accelerator.backward(alpha_loss)
        self.log_alpha_optimizer.step()

        self.soft_update(self.actors[1], self.target_critic_1)
        self.soft_update(self.actors[2], self.target_critic_2)
        
        return actor_loss.item(), critic_1_loss.item(), critic_2_loss.item(), alpha_loss.item()

    def train_one_epoch(self, loader, replay_buffer, minimal_size, epoch, loader_attributes):
        total_loss          = 0.0
        total_return        = 0.0
        total_improvement   = 0.0
        num_samples         = 0
        self.loss_iter_counter = self.epoch * len(loader)
        with tqdm(total=len(loader), desc='Training') as pbar:
            for data in loader:
                ######## preparation
                preds = []
                episode_return = 0
                # data = next(loader_iter)
                # if self.move_data_to_gpu:
                #     data = data.to(self.device)
                # if self.accelerator.is_main_process:
                #     print(f"meta info: {data['meta_info']}")
                self.epoch = epoch
                data['epoch'] = self.epoch
                data['settings'] = self.settings
                state = data['burst']
                meta_info = data['meta_info']
                
                done = 0
                with torch.no_grad():
                    pred, _   = self.sr_net(state)
                preds.append(pred.clone())
                batch_size = data['frame_gt'].size(0)
                if self.accelerator.is_main_process:
                    print(f"Data frame gt batch size {batch_size}")
                permutations = torch.tensor(self.init_permutation, device=self.accelerator.device).repeat(batch_size, 1, 1)

                if self.reward_type == 'psnr':
                    reward_func = {'psnr': PSNR(boundary_ignore=40)}
                elif self.reward_type == 'ssim':
                    reward_func = {'ssim': SSIM(boundary_ignore=40)}
                else:
                    assert 0 == 1, "wrong reward type"
                ######## preparation done 
                
                iteration = 0
                actor_loss_episode, critic_1_loss_episode, critic_2_loss_episode, alpha_loss_episode = 0, 0, 0, 0
                if self.accelerator.is_main_process:
                    initial_permute = permutations.clone()
                    # print(f"Initial permutea at loss_iter_counter {self.loss_iter_counter}: {initial_permute}")
                while not done:
                    # action = agent.take_action(state)
                    # next_state, reward, done, _ = env.step(action)
                    
                    # substitute the above two lines with below line
                    probs = self.actors[0](state) # here the state should be one
                    dists = [Categorical(p) for p in probs.split(1, dim=1)]
                    next_state, action, permutations, done = self.step_environment(dists, data['frame_gt'].clone(), permutations.clone(), iteration, add_noise=True, meta_info=meta_info)
                    if self.accelerator.is_main_process:
                        inter_permute = permutations.clone()
                        # print(f"Inter permutea at loss_iter_counter {self.loss_iter_counter} in timestep {iteration}: {inter_permute}")
                    with torch.no_grad():
                        # print("device of model: ", next(self.sr_net.parameters()).device)
                        # print("device of nextstate: ", next_state.size())
                        pred, _ = self.sr_net(next_state.clone())
                    preds.append(pred.clone())
                    reward = self._calculate_reward(data['frame_gt'], preds[-1], preds[-2], reward_func=reward_func, batch=True, tolerance=0)
                    # if iteration <= 5 or done:
                    replay_buffer.add(state.cpu().clone(), action.cpu().clone(), reward.cpu().clone(), next_state.cpu().clone(), done)
                    # if self.accelerator.is_main_process:
                    #     print(f"[Main GPU] Replay buffer size: {len(replay_buffer.buffer)}")
                    # else:
                    #     # 假设有一种方法来获取当前GPU的ID，例如 `current_gpu_id`
                    #     current_gpu_id = torch.cuda.current_device()
                    #     print(f"[GPU-{current_gpu_id}] Replay buffer size: {len(replay_buffer.buffer)}")

                    # print(f"Debug what is the size of state: {state.size()}")
                    # print(f"Debug what is the size of action: {action.size()}")
                    # print(f"Debug what is the size of reward: {reward.size()}")
                    # print(f"Debug what is the size of next_state: {next_state.size()}")
                    # print(f"Debug what is the size of done: {done}")
                    state = next_state.clone()
                    episode_return += reward * (self.discount_factor ** iteration)
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(self.sample_size)
                        current_gpu_id = torch.cuda.current_device()
                        # print(f"[GPU-{current_gpu_id}] Sampled data size: {len(b_s)}")

                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        actor_loss, critic_1_loss, critic_2_loss, alpha_loss = self.update(transition_dict)
                        actor_loss_episode += actor_loss
                        critic_1_loss_episode += critic_1_loss
                        critic_2_loss_episode += critic_2_loss
                        alpha_loss_episode += alpha_loss
                        # self._update_stats({'Loss/actor_loss': actor_loss, 'Loss/critic_1_loss': critic_1_loss, \
                        #         'Loss/critic_2_loss': critic_2_loss, "Loss/alpha_loss": alpha_loss, \
                        #         }, batch_size, loader_attributes)
                    iteration += 1
                if self.accelerator.is_main_process:
                    final_permute = permutations.clone()
                    # print(f"Final permutea at loss_iter_counter {self.loss_iter_counter}: {final_permute}")
                actor_loss_episode = actor_loss_episode / iteration
                critic_1_loss_episode = critic_1_loss_episode / iteration
                critic_2_loss_episode = critic_2_loss_episode / iteration
                alpha_loss_episode = alpha_loss_episode / iteration
                if self.accelerator.is_main_process:
                    self.writer.add_scalar('Training Loss/actor_loss', actor_loss_episode, self.loss_iter_counter)
                    self.writer.add_scalar('Training Loss/critic_1_loss', critic_1_loss_episode, self.loss_iter_counter)
                    self.writer.add_scalar('Training Loss/critic_2_loss', critic_2_loss_episode, self.loss_iter_counter)
                    self.writer.add_scalar('Training Loss/alpha_loss', alpha_loss_episode, self.loss_iter_counter)
                metric_initial = reward_func[self.reward_type](preds[0].clone(), data['frame_gt'].clone())
                metric_final = reward_func[self.reward_type](preds[-1].clone(), data['frame_gt'].clone())
                total_loss += actor_loss_episode + critic_1_loss_episode + critic_2_loss_episode + alpha_loss_episode
                total_return += episode_return.mean().item()
                num_samples += 1 
                improvement = metric_final.item() - metric_initial.item()
                if self.accelerator.is_main_process:
                    self.writer.add_scalar('Training Loss/improvement', improvement, self.loss_iter_counter)
                    self.writer.add_scalar('Training Loss/inital_psnr', metric_initial.item(), self.loss_iter_counter)
                    self.writer.add_scalar('Training Loss/final_psnr', metric_final.item(), self.loss_iter_counter)
                total_improvement += improvement
                self.loss_iter_counter += 1
                if self.accelerator.is_main_process:
                    pbar.set_postfix({'loss': total_loss / num_samples, 'return': total_return / num_samples, 'Improvement': improvement})
                    pbar.update(1)
                
        return total_loss / num_samples, total_return / num_samples, total_improvement / num_samples

    def validate_one_epoch(self, loader, loader_attributes):
        total_improvement   = 0.0
        num_samples         = 0
        with tqdm(total=len(loader), desc='Validation') as pbar:
            for idx, data in enumerate(loader, 1):
                # if self.move_data_to_gpu:
                #     data = data.to(self.accelerator.device)
                state = data['burst']
                done = 0
                with torch.no_grad():
                    pred, _   = self.sr_net(state)
                if self.save_results:
                    initial_pred_meta_info = data['meta_info']
                    burst_rgb = data['burst_rgb'].clone()
                initial_pred = pred.clone()
                final_pred  =pred.clone()
                batch_size = data['frame_gt'].size(0)
                permutations = torch.tensor(self.init_permutation, device=self.accelerator.device).repeat(batch_size, 1, 1)
                if self.reward_type == 'psnr':
                    reward_func = {'psnr': PSNR(boundary_ignore=40)}
                elif self.reward_type == 'ssim':
                    reward_func = {'ssim': SSIM(boundary_ignore=40)}
                else:
                    assert 0 == 1, "wrong reward type"
                meta_info = data['meta_info']
                ######## preparation done 
                iteration = 0
                while not done:
                    with torch.no_grad():
                        probs = self.actors[0](state) # here the state should be one
                    dists = [Categorical(p) for p in probs.split(1, dim=1)]
                    next_state, action, permutations, done = self.step_environment(dists, data['frame_gt'].clone(), permutations.clone(), iter=iteration, add_noise=True, meta_info=meta_info, save_results=self.save_results)
                    with torch.no_grad():
                        # print("device of model: ", next(self.sr_net.parameters()).device)
                        # print("device of nextstate: ", next_state.size())
                        pred, _ = self.sr_net(next_state.clone())
                    final_pred = pred.clone()
                    state = next_state.clone()
                    iteration += 1

                metric_initial = reward_func[self.reward_type](initial_pred.clone(), data['frame_gt'].clone())
                metric_final = reward_func[self.reward_type](final_pred.clone(), data['frame_gt'].clone())
                improvement = metric_final.item() - metric_initial.item()
                total_improvement += improvement
                num_samples += 1
                if self.accelerator.is_main_process:
                    pbar.set_postfix({'improvement': improvement})
                    pbar.update(1)
                if self.save_results:
                    # save vis results
                    self.save_img_and_metrics(initial_pred=initial_pred.clone(), final_pred=final_pred.clone(), \
                        initial_psnr=metric_initial.item(), final_psnr=metric_final.item(), meta_info=initial_pred_meta_info, \
                            burst_rgb=burst_rgb, gt=data['frame_gt'].clone(), final_shifts=permutations.clone(), name=data['image_name'])
                    # save trajectories
                    f=open("%s/traj.pkl" % self.saving_dir, 'wb')
                    pickle.dump(self.final_permutations, f)
                    f.close()
                    f=open("%s/traj_int_length_%s.pkl" % (self.saving_dir, self.one_step_length), 'wb')
                    pickle.dump(self.final_permutations_int_length, f)
                    f.close()
                    # save metrics
                    f=open("%s/metrics.txt" % self.saving_dir, 'a')
                    print("%sth psnr intial: %s, final: %s, improvement: %s | Average psnr initial: %s, final: %s, improvement: %s" % \
                            (data['image_name'], metric_initial.item(), metric_final.item(), (metric_final.item()-metric_initial.item()), \
                                self.initial_psnr_sum/idx, self.final_psnr_sum/idx, \
                                    self.final_psnr_sum/idx - self.initial_psnr_sum/idx), file=f)
                    f.close()
        return total_improvement / num_samples      

    def save_replay_buffer(self, replay_buffer, filename):
        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_file = os.path.join(directory, filename)
        with open(save_file, 'wb') as f:
            pickle.dump(replay_buffer, f)

    def train_off_policy_agent(self, loaders, max_epochs, replay_buffer, minimal_size=50):
        return_list = []
        for epoch in range(self.epoch, max_epochs):
            for i_loader, loader in enumerate(loaders):
                if self.epoch % self.loader_attributes[i_loader]['epoch_interval'] == 0:
                    if self.loader_attributes[i_loader]['training']:
                        print("Current Training Epoch is %s" % self.epoch)
                        avg_loss, avg_return, avg_improvement = self.train_one_epoch(loader, replay_buffer, minimal_size, epoch=epoch, loader_attributes=self.loader_attributes[i_loader])
                        # Optionally log or print avg_loss and avg_return
                        
                        if self.accelerator.is_main_process:
                            self.writer.add_scalar('Training Loss', avg_loss, self.epoch)
                            self.writer.add_scalar('Training Return', avg_return, self.epoch)
                            self.writer.add_scalar('Training Improvement', avg_improvement, self.epoch)
                    else:
                        print("Current Validating Epoch is %s" % self.epoch)
                        avg_improvement = self.validate_one_epoch(loader, loader_attributes=self.loader_attributes[i_loader])
                        if self.save_results:
                            print("This is validation mode! Done.")
                            exit()
                        if self.accelerator.is_main_process:
                            self.writer.add_scalar('Testing Improvement', avg_improvement, self.epoch)
                        if self.accelerator.is_main_process:
                            print(f"Starting saving checkpoint!!!!!!")
                            self.save_checkpoint()
                            print(f"Completed!!!!!!!!!!")
                            # if replay_buffer.size() < 100:
                            #     self.save_replay_buffer(replay_buffer, 'replay_buffer.pkl')
                        # Optionally log or print avg_improvement
                    self._stats_new_epoch()
                    # self._write_tensorboard()
                    self.epoch += 1
        return return_list
