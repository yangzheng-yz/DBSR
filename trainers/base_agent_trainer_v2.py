# this version requires to listed actors
import os
import glob
import torch
import traceback
from admin import loading, multigpu


class BaseAgentTrainer:
    """Base trainer class. Contains functions for training and saving/loading chackpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actors, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        self.actors = actors
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        # print("basetrainer's device: ", self.device)
        for idx, _ in enumerate(self.actors): 
            self.actors[idx].to(self.device)

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

    def train(self, max_epochs, load_latest=False, fail_safe=True, checkpoint = None):
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
                

                for epoch in range(self.epoch+1, max_epochs+1):
                    self.epoch = epoch

                    self.train_epoch()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    if self._checkpoint_dir:
                        self.save_checkpoint()
            except:
                print('Training crashed at epoch {}'.format(epoch))
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

    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""

        # net = self.actor.module if multigpu.is_multi_gpu(self.actor) else self.actor
        nets = self.actors
        print("Temporarily we do not support multigpu.")
        
        actors_type = [type(actor).__name__ for actor in self.actors]
        nets_type = [type(net).__name__ for net in nets]
        states = [{
            'epoch': self.epoch,
            'actor_type': actors_type[idx],
            'net_type': nets_type[idx],
            'net': nets[idx].state_dict(),
            'net_info': getattr(nets[idx], 'info', None),
            'constructor': getattr(nets[idx], 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
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

        # net = self.actor.module if multigpu.is_multi_gpu(self.actor) else self.actor
        nets = self.actors
        print("Temporarily we do not support multigpu. ")
        
        actors_type = [type(actor).__name__ for actor in self.actors]
        nets_type = [type(net).__name__ for net in nets]

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
        elif isinstance(checkpoint, list):
            checkpoints_path = [os.path.expanduser(checkp) for checkp in checkpoint]
        else:
            raise TypeError

        # Load network
        checkpoints_dict = [torch.load(checkpoint_path) for checkpoint_path in checkpoints_path]

        for idx, net_type in enumerate(nets_type):
            assert net_type == checkpoints_dict[idx]['net_type'], 'Network [%s] is not of correct type.' % net_type

        if fields is None:
            fields = checkpoints_dict[0].keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

            # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                for idx, _ in enumerate(nets):
                    nets[idx].load_state_dict(checkpoints_dict[idx][key])
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoints_dict[0][key])
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
            self.lr_scheduler.last_epoch = self.epoch

        return True
