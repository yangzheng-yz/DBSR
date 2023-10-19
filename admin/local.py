class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/mnt/samsung/zheng/downloaded_datasets/zheng_ccvl21/training_log'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_nets_dir = '/home/yutong/zheng/projects/dbsr_rl/mice_train_ccvl6/admin/../pretrained_networks'    # Directory for pre-trained networks.
        self.save_data_path = ''    # Directory for saving network predictions for evaluation.
        self.zurichraw2rgb_dir = '/mnt/samsung/zheng/downloaded_datasets/SyntheticDataset'    # Zurich RAW 2 RGB path
        self.burstsr_dir = ''    # BurstSR dataset path
        self.synburstval_dir = ''    # SyntheticBurst validation set path
        self.MixedMiceNIR_Dai_dir = '/mnt/samsung/zheng/downloaded_datasets/NIRII'
