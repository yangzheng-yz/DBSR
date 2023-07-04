import dataset as datasets
from data import processing, sampler, DataLoader
import data.transforms as tfm
import os
import numpy as np
import cv2
import pickle as pkl
import argparse
import tqdm
import yaml
from easydict import EasyDict
import torch

from evaluation.common_utils.network_param import NetworkParam
from models.loss.image_quality_v2 import PSNR, SSIM, LPIPS
from data.postprocessing_functions import SimplePostProcess

cfg = EasyDict()

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--ckpt_path', type=str, default=None, help='checkpoint to be evaluated')
    parser.add_argument('--dataset_path', type=str, default='/mnt/samsung/zheng/downloaded_datasets/NightCity_1024x512/val', help='dataset to be evaluated')
    parser.add_argument('--trajectory_path', type=str, default=None, help='to specify the pixel shift trajectory, the dir should include a .txt(trajectories) and a .pkl(meta_info)')
    parser.add_argument('--save_path', type=str, default=None, help='dir to save all output')    
    parser.add_argument('--use_saved_results', type=str, default=None, help='use previous predicted results path, no need GPU')    

    parser.add_argument('--save_results', action='store_true', default=True, help='save the superresolution SR, the first LR, and HR in sRGB')
    parser.add_argument('--save_pixelShifts', action='store_true', default=False, help='save the pixel shifted LRs and HR in linear sensor space')    
    
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    
    return args, cfg

def main():
    args, cfg = parse_config()
    """The first part is to prepare the dataset and define the evaluation metrics"""
    assert args.dataset_path is not None, "You must specify the dataset path"
    NightCity_val = datasets.NightCity(root=args.dataset_path, split='val')   
    
    metrics = ('psnr', 'ssim')
    device = 'cuda'
    boundary_ignore = 40
    metrics_all = {}
    scores = {}
    for m in metrics:
        if m == 'psnr':
            loss_fn = PSNR(boundary_ignore=boundary_ignore)
        elif m == 'ssim':
            loss_fn = SSIM(boundary_ignore=boundary_ignore, use_for_loss=False)
        elif m == 'lpips':
            loss_fn = LPIPS(boundary_ignore=boundary_ignore)
            loss_fn.to(device)
        else:
            raise Exception
        metrics_all[m] = loss_fn
        scores[m] = []

    scores_all = {}
    
    """The second part is to load the trained checkpoints"""
    assert args.ckpt_path is not None, "You must specify a pretrained weights to evaluate."
    n = NetworkParam(network_path='%s' % args.ckpt_path, # both .pth and .pth.tar can be loaded 
                                     unique_name='%s' % args.ckpt_path.split('/')[-2])         # Unique name is used when saving results

    using_saved_results = False
    
    if args.use_saved_results is not None:
        # Check if results directory exists
        if os.path.isdir(args.use_saved_results):
            result_list = os.listdir(args.use_saved_results)
            result_list = [res for res in result_list if res[-3:] == 'png']

            # Check if number of results match
            # TODO use a better criteria
            if len(result_list) == len(dataset):
                using_saved_results = True

    
    if not using_saved_results:
        net = n.load_net()
        device = 'cuda'
        net.to(device).train(False) 
    
    """The third part is to define the transformation's type"""
    assert args.trajectory_path is not None, "You must generate a trajectory.txt firstly to perform this pixel shift evaluation."
    with open(args.trajectory_path, 'rb') as f:
        permutations = pkl.load(f)
    
    scores_all_mean = {}
    for idx_traj, permutation in enumerate(permutations):
        print("processing %sth trajectory of %s" % (idx_traj, args.trajectory_path))
        transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())
        image_processing_params = {'random_ccm': cfg.random_ccm, 'random_gains': cfg.random_gains, 'smoothstep': cfg.smoothstep, 'gamma': cfg.gamma, 'add_noise': cfg.add_noise}
        burst_transformation_params_val = {'max_translation': 3.0,
                                            'max_rotation': 0.0,
                                            'max_shear': 0.0,
                                            'max_scale': 0.0,
                                            'border_crop': 24,
                                            'random_pixelshift': False,
                                            'specified_translation': permutation}
        data_processing_val = processing.SyntheticBurstDatabaseProcessing((cfg.crop_sz, cfg.crop_sz), cfg.burst_sz,
                                                                            cfg.downsample_factor,
                                                                            burst_transformation_params=burst_transformation_params_val,
                                                                            transform=transform_val,
                                                                            image_processing_params=image_processing_params,
                                                                            random_crop=False,
                                                                            return_rgb_busrt=cfg.return_rgb_busrt)
        dataset_val = sampler.IndexedImage(NightCity_val, processing=data_processing_val)
        process_fn = SimplePostProcess(return_np=True)

        """The fourth part is to perform prediction"""
        for idx, data in enumerate(dataset_val):
            print("evaluated %s/%s images of %s" % (idx, len(dataset_val)-1, args.dataset_path))
            burst = data['burst']
            gt = data['frame_gt']
            meta_info = data['meta_info']
            burst_rgb = data['burst_rgb']
            assert cfg.return_rgb_burst, "Better open this button to save the results."
            meta_info['frame_num'] = idx
            burst_name = data['image_name']

            burst = burst.to(device).unsqueeze(0)
            gt = gt.to(device)

            if n.burst_sz is not None:
                burst = burst[:, :n.burst_sz]

            if using_saved_results:
                net_pred = cv2.imread('{}/{}_SR.png'.format(args.use_saved_results, burst_name), cv2.IMREAD_UNCHANGED)
                net_pred = (torch.from_numpy(net_pred.astype(np.float32)) / 2 ** 14).permute(2, 0, 1).float().to(device)
                net_pred = net_pred.unsqueeze(0)
            else:
                with torch.no_grad():
                    net_pred, _ = net(burst)

                # Perform quantization to be consistent with evaluating on saved images
                net_pred_int = (net_pred.clamp(0.0, 1.0) * 2 ** 14).short()
                net_pred = net_pred_int.float() / (2 ** 14)

            for m, m_fn in metrics_all.items():
                metric_value = m_fn(net_pred, gt.unsqueeze(0)).cpu().item()
                scores[m].append(metric_value)
                
            # Here we want to save result for visualization
            if args.save_results and args.save_path is not None:
                if not os.path.isdir(args.save_path):
                    os.makedirs('{}'.format(args.save_path), exist_ok=True)
                    
                save_path_traj = os.path.join(args.save_path, '{:04d}' % idx_traj)
                if not os.path.isdir(save_path_traj):
                    os.makedirs(save_path_traj, exist_ok=True)
                
                HR_image = process_fn.process(gt.cpu(), meta_info)
                LR_image = process_fn.process(burst_rgb[0], meta_info)
                SR_image = process_fn.process(net_pred.cpu(), meta_info)
                
                HR_image = cv2.resize(HR_image, dsize=(gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                LR_image = cv2.resize(LR_image, dsize=(gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                SR_image = cv2.resize(SR_image, dsize=(gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                HR_image = (HR_image.permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).numpy().astype(np.uint16)
                LR_image = (LR_image.permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).numpy().astype(np.uint16)
                SR_image = (SR_image.permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).numpy().astype(np.uint16)
                
                cv2.imwrite('{}/HR_{}.png'.format(save_path_traj, burst_name), HR_image)
                cv2.imwrite('{}/LR_{}.png'.format(save_path_traj, burst_name), LR_image)
                cv2.imwrite('{}/SR_{}.png'.format(save_path_traj, burst_name), SR_image)
                
        # scores_all[n.get_display_name()] = scores
        scores_all_mean['%s_%sth-Traj'] = {m: sum(s) / len(s) for m, s in scores.items()}
    with open(os.path.join(args.save_path, 'results_of_%s-%s.pkl' % (args.ckpt_path.split('/')[-1].split('.')[0], args.trajectory_path.split('/')[-1].split('.')[0])), 'wb') as f:
        pkl.dump(scores_all_mean, f)


if __name__ == '__main__':
    main()
