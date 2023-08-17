class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/mnt/samsung/zheng/downloaded_datasets/zheng_ccvl2/training_results'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_nets_dir = '/home/yutong/zheng/projects/DBSR/pretrained_networks'    # Directory for pre-trained networks.
        self.save_data_path = '/home/yutong/zheng/projects/DBSR/synburst'    # Directory for saving network predictions for evaluation.
        self.zurichraw2rgb_dir = '/mnt/samsung/zheng/downloaded_datasets/zheng_ccvl2/SyntheticDataset'    # Zurich RAW 2 RGB paths
        self.nightcity_dir = '/mnt/samsung/zheng/downloaded_datasets/zheng_ccvl2/SyntheticDataset/NightCity_1024x512/'
        self.burstsr_dir = '/mnt/samsung/zheng/downloaded_datasets/zheng_ccvl2/SyntheticDataset/real_world/burstsr_dataset'    # BurstSR dataset path
        self.synburstval_dir = '/mnt/samsung/zheng/downloaded_datasets/zheng_ccvl2/SyntheticDataset/val'    # SyntheticBurst validation set path
        self.nir_synthetic_dir = '/mnt/samsung/zheng/downloaded_datasets/zheng_ccvl2/NIR-II/MOESM'
        self.nir_visible_dir = '/mnt/samsung/zheng/downloaded_datasets/zheng_ccvl2/nir_visible'