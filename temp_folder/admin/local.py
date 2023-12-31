class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_nets_dir = '/home/zheng/projects/deep-burst-sr/admin/../pretrained_networks/'    # Directory for pre-trained networks.
        self.save_data_path = ''    # Directory for saving network predictions for evaluation.
        self.zurichraw2rgb_dir = ''    # Zurich RAW 2 RGB path
        self.burstsr_dir = ''    # BurstSR dataset path
        self.synburstval_dir = ''    # SyntheticBurst validation set path
