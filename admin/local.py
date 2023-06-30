class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/mnt/samsung/zheng/downloaded_datasets/NightCity_1024x512/training_results'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_nets_dir = '/home/yutong/zheng/projects/DBSR/admin/../pretrained_networks/'    # Directory for pre-trained networks.
        self.save_data_path = '/home/yutong/zheng/projects/DBSR/synburst'    # Directory for saving network predictions for evaluation.
        self.zurichraw2rgb_dir = ''    # Zurich RAW 2 RGB paths
        self.nightcity_dir = '/mnt/samsung/zheng/downloaded_datasets/NightCity_1024x512/'
        self.burstsr_dir = ''    # BurstSR dataset path
        self.synburstval_dir = '/home/yutong/zheng/projects/DBSR/downloaded_datasets/SyntheticBurstVal'    # SyntheticBurst validation set path
