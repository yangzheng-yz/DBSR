# this version requires to listed actors
import os
import glob
import torch
import traceback
from admin import loading, multigpu
import numpy as np
from utils.rl_utils import ReplayBuffer
import pickle

class BaseAgentTrainer:
    """Base trainer class. Contains functions for training and saving/loading chackpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actors, loaders, actor_optimizer, critic_1_optimizer, critic_2_optimizer, log_alpha_optimizer, settings, actor_lr_scheduler=None, critic_1_lr_scheduler=None, critic_2_lr_scheduler=None, log_alpha_lr_scheduler=None, log_alpha=0, actors_attr=None, inital_epoch=0):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            high_level_lr_scheduler - Learning rate scheduler
        """
        self.actors_attr = actors_attr
        self.actors = actors
        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer
        self.actor_lr_scheduler = actor_lr_scheduler
        self.critic_1_lr_scheduler = critic_1_lr_scheduler
        self.critic_2_lr_scheduler = critic_2_lr_scheduler

        self.log_alpha = log_alpha
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = log_alpha_optimizer
        self.log_alpha_lr_scheduler = log_alpha_lr_scheduler
        
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = inital_epoch
        self.stats = {}

        # self.device = getattr(settings, 'device', None)
        # if self.device is None:
        #     self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        # print("basetrainer's device: ", self.device)
        # for idx, _ in enumerate(self.actors): 
        #     self.actors[idx].to(self.device)

    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    def load_replay_buffer(self, filename):
        with open(filename, 'rb') as f:
            replay_buffer = pickle.load(f)
        return replay_buffer

    def train(self, max_epochs, load_latest=False, fail_safe=True, checkpoint = None, buffer_size=10000):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
        """

        epoch = -1
        num_tries = 1
        for i in range(num_tries):
            try:
                if load_latest:
                    self.load_checkpoint()
                else:
                    if isinstance(checkpoint, str) or isinstance(checkpoint, list):
                        self.load_checkpoint(checkpoint)
                directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                if os.path.exists(os.path.join(directory, 'replay_buffer.pkl')):
                    replay_buffer = self.load_replay_buffer(os.path.join(directory, 'replay_buffer.pkl'))
                    print(f"Successfully load last replay buffer with size {replay_buffer.size()}")
                else:
                    replay_buffer = ReplayBuffer(buffer_size)
                    print(f"Successfully initialize replay buffer with size {replay_buffer.size()}")
                self.train_sac(max_epochs, replay_buffer)
                # for epoch in range(self.epoch+1, max_epochs+1):
                #     self.epoch = epoch

                    # self.train_epoch()

                    # if self.high_level_lr_scheduler is not None:
                    #     self.high_level_lr_scheduler.step()
                    # if self.option_lr_scheduler is not None:
                    #     self.option_lr_scheduler.step()

                    # if self._checkpoint_dir:
                    #     self.save_checkpoint()
            except:
                print('Training crashed at epoch {}'.format(self.epoch))
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')

    def train_epoch(self):
        raise NotImplementedError

    def train_sac(self):
        raise NotImplementedError

    # def save_checkpoint(self):
    #     """Saves a checkpoint of the network and other variables."""

    #     # nets = [actor.module if multigpu.is_multi_gpu(actor) else actor for actor in self.actors]
    #     nets = self.actors
    #     # print("Temporarily we do not support multigpu.")
        
    #     actors_type = [f"{type(actor).__name__}_{idx}" for idx, actor in enumerate(self.actors)]
    #     nets_type = [f"{type(net).__name__}_{idx}" for idx, net in enumerate(nets)]
    #     states = [{
    #         'epoch': self.epoch,
    #         'actor_type': actors_type[idx],
    #         'net_type': nets_type[idx],
    #         'net': nets[idx].state_dict(),
    #         'net_info': getattr(nets[idx], 'info', None),
    #         'constructor': getattr(nets[idx], 'constructor', None),
    #         'actor_optimizer': self.actor_optimizer.state_dict(),
    #         'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
    #         'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
    #         'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),
    #         'stats': self.stats,
    #         'settings': self.settings
    #     } for idx, _ in enumerate(nets_type)]

    #     directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
        
    #     for net_type in nets_type:
    #         if not os.path.exists(os.path.join(directory, net_type)):
    #             os.makedirs(os.path.join(directory, net_type))

    #     # First save as a tmp file
    #     tmp_files_path = ['{}/{}/ep{:04d}.tmp'.format(directory, nets_type[idx], self.epoch) for idx, _ in enumerate(nets_type)]
    #     for idx, tmp_file_path in enumerate(tmp_files_path):
    #         torch.save(states[idx], tmp_file_path)

    #         file_path = '{}/{}/ep{:04d}.pth.tar'.format(directory, nets_type[idx], self.epoch)

    #         # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
            
    #         os.rename(tmp_file_path, file_path)

    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""

        print(f"what is now processing {self.accelerator.is_main_process}")
        if not self.accelerator.is_main_process:
            return
        nets = [self.accelerator.unwrap_model(actor) for actor in self.actors]
        
        actors_type = self.actors_attr
        nets_type = self.actors_attr
        states = [{
            'epoch': self.epoch,
            'actor_type': actors_type[idx],
            'net_type': nets_type[idx],
            'net': nets[idx].state_dict(),
            'net_info': getattr(nets[idx], 'info', None),
            'constructor': getattr(nets[idx], 'constructor', None),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
            'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha.item(),
            'stats': self.stats,
            'settings': self.settings
        } for idx, _ in enumerate(nets_type)]

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for net_type in nets_type:
            if not os.path.exists(os.path.join(directory, net_type)):
                os.makedirs(os.path.join(directory, net_type))

        # First save as a tmp file
        tmp_files_path = ['{}/{}/ep{:04d}.tmp'.format(directory, nets_type[idx], self.epoch) for idx, _ in enumerate(nets_type)]
        for idx, tmp_file_path in enumerate(tmp_files_path):
            torch.save(states[idx], tmp_file_path)

            file_path = '{}/{}/ep{:04d}.pth.tar'.format(directory, nets_type[idx], self.epoch)

            # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
            
            os.rename(tmp_file_path, file_path)

    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        # nets preparation for loading
        nets = [actor.module if multigpu.is_multi_gpu(actor) else actor for actor in self.actors]
        nets_type = [f"{type(net).__name__}_{idx}" for idx, net in enumerate(nets)]
        
        # actors_type = [f"{type(actor).__name__}_{idx}" for idx, actor in enumerate(self.actors)]
        nets_type = [f"{type(net).__name__}_{idx}" for idx, net in enumerate(nets)]

        if self.accelerator.is_main_process:
            if checkpoint is None:
                # Load most recent checkpoint
                checkpoints_list = [sorted(glob.glob('{}/{}/{}/ep*.pth.tar'.format(self._checkpoint_dir,
                                                                                self.settings.project_path, net_type))) for net_type in nets_type]
                checkpoints_path = []
                for idx, checkpoint_list in enumerate(checkpoints_list):
                    if checkpoint_list:
                        checkpoints_path.append(checkpoint_list[-1])
                    else:
                        print('No matching checkpoint file found[%s]' % nets_type[idx])
                        return
            elif isinstance(checkpoint, int):
                # Checkpoint is the epoch number
                checkpoints_path = ['{}/{}/{}/ep{:04d}.pth.tar'.format(self._checkpoint_dir, self.settings.project_path,
                                                                    net_type, checkpoint) for net_type in nets_type]
            elif isinstance(checkpoint, str):
                # checkpoint is the path
                if os.path.isdir(checkpoint):
                    checkpoints_list = [sorted(glob.glob('{}/{}/ep*.pth.tar'.format(checkpoint, net_type))) for net_type in nets_type]
                    checkpoints_path = []
                    for idx, checkpoint_list in enumerate(checkpoints_list):
                        if checkpoint_list:
                            checkpoints_path.append(checkpoint_list[-1])
                        else:
                            raise Exception('No checkpoint found[%s]' % nets_type[idx])
                else:
                    checkpoint_path = os.path.expanduser(checkpoint)
            elif isinstance(checkpoint, list):
                checkpoints_path = [os.path.expanduser(checkp) for checkp in checkpoint]
            else:
                raise TypeError

            # Load network
            if isinstance(checkpoint_path, list):
                checkpoints_dict = [torch.load(checkpoint_path) for checkpoint_path in checkpoints_path]
            else:
                checkpoints_dict = torch.load(checkpoint_path)
        else:
            checkpoints_dict = [None] * len(nets_type)
        
        if isinstance(checkpoints_dict, list):
            checkpoints_dict = [self.accelerator.broadcast(checkpoint) for checkpoint in checkpoints_dict]
        else:
            checkpoints_dict = self.accelerator.broadcast(checkpoints_dict)

        for idx, net_type in enumerate(nets_type):
            assert net_type == checkpoints_dict[idx]['net_type'], 'Network [%s] is not of correct type.' % net_type

        if fields is None:
            fields = checkpoints_dict[0].keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

            # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['high_level_lr_scheduler', 'option_lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                for idx, _ in enumerate(nets):
                    nets[idx].load_state_dict(checkpoints_dict[idx][key])
            elif key == 'actor_optimizer':
                self.actor_optimizer.load_state_dict(checkpoints_dict[0][key])
            elif key == 'critic_1_optimizer':
                self.critic_1_optimizer.load_state_dict(checkpoints_dict[0][key])
            elif key == 'critic_2_optimizer':
                self.critic_2_optimizer.load_state_dict(checkpoints_dict[0][key])
            elif key == 'log_alpha_optimizer':
                self.log_alpha_optimizer.load_state_dict(checkpoints_dict[0][key])
            else:
                # print("what are the keys: ", checkpoints_dict[0].keys())
                # print("what is current key: ", key)
                setattr(self, key, checkpoints_dict[0][key])

        # Set the net info
        if load_constructor and 'constructor' in checkpoints_dict[0] and checkpoints_dict[0]['constructor'] is not None:
            for idx, _ in enumerate(nets):
                nets[idx].constructor = checkpoints_dict[idx]['constructor']
        if 'net_info' in checkpoints_dict[0] and checkpoints_dict[0]['net_info'] is not None:
            for idx, _ in enumerate(nets):
                nets[idx].info = checkpoints_dict[idx]['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.actor_lr_scheduler.last_epoch = self.epoch
            self.critic_1_lr_scheduler.last_epoch = self.epoch
            self.critic_2_lr_scheduler.last_epoch = self.epoch
            self.log_alpha_lr_scheduler.last_epoch = self.epoch

        return True
