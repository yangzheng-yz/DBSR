class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/mnt/data0/zheng/training_results'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_nets_dir = '/home/yutong/zheng/projects/dbsr_us/pretrained_networks'    # Directory for pre-trained networks.
        self.save_data_path = '/home/yutong/zheng/projects/dbsr_us/synburst'    # Directory for saving network predictions for evaluation.
        self.zurichraw2rgb_dir = '/mnt/data0/zheng/SyntheticDataset'    # Zurich RAW 2 RGB paths
        self.nightcity_dir = '/mnt/data0/zheng/SyntheticDataset/NightCity_1024x512/'
        self.burstsr_dir = '/mnt/data0/zheng/SyntheticDataset/real_world/burstsr_dataset'    # BurstSR dataset path
        self.synburstval_dir = '/mnt/data0/zheng/SyntheticDataset/val'    # SyntheticBurst validation set path
        self.nir_synthetic_dir = '/mnt/data0/zheng/NIR-II/MOESM'
        self.nir_visible_dir = '/mnt/data0/zheng/nir_visible'