"""
This version is for the nir visible dataset.
"""
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
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import io, img_as_float
from sys import argv

from evaluation.common_utils.network_param import NetworkParam
from models_dbsr.loss.image_quality_v2 import PSNR, SSIM, LPIPS
from data.postprocessing_functions import SimplePostProcess

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

cfg = EasyDict()

def calculate_psnr(img1, img2):

    # Convert the images to float
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)

    # Calculate the PSNR
    psnr_value = psnr(img1, img2, data_range=img1.max() - img1.min())

    return psnr_value

def calculate_ssim(img1, img2):
    # Convert the images to float
    img1 = img_as_float(img1)
    img2 = img_as_float(img2)

    # Calculate the SSIM
    ssim_value = ssim(img1, img2, multichannel=True)

    return ssim_value

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

    cfg_from_yaml_file(argv[1], cfg)
    
    return cfg

def main():
    cfg = parse_config()
    """The first part is to prepare the dataset and define the evaluation metrics"""
    assert cfg.dataset_path is not None, "You must specify the dataset path"
    nir_visible_val = datasets.nir_visible(burst_sz=cfg.burst_sz, split='test')
    
    metrics = ('psnr', 'ssim', 'lr_psnr', 'lr_ssim')
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
        elif m == 'lr_psnr':
            loss_fn = PSNR(boundary_ignore=boundary_ignore)
        elif m == 'lr_ssim':
            loss_fn = SSIM(boundary_ignore=boundary_ignore, use_for_loss=False)
        else:
            raise Exception
        metrics_all[m] = loss_fn
        scores[m] = []

    scores_all = {}
    
    """The second part is to load the trained checkpoints"""
    assert cfg.ckpt_path is not None, "You must specify a pretrained weights to evaluate."
    n = NetworkParam(network_path='%s' % cfg.ckpt_path, # both .pth and .pth.tar can be loaded 
                                     unique_name='%s' % cfg.ckpt_path.split('/')[-2])         # Unique name is used when saving results

    using_saved_results = False
    
    if cfg.use_saved_results is not None:
        # Check if results directory exists
        if os.path.isdir(cfg.use_saved_results):
            result_list = os.listdir(cfg.use_saved_results)
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
    assert cfg.trajectory_path is not None, "You must generate a trajectory.pkl firstly to perform this pixel shift evaluation."
    with open(cfg.trajectory_path, 'rb') as f:
        permutations = pkl.load(f)
    
    scores_all_mean = {}
    selected_images_id = np.arange(0, 100000) 
    
    for idx_traj, permutation in enumerate(permutations):
        if cfg.specify_trajetory_num != -1:
            if idx_traj != cfg.specify_trajetory_num:
                continue
        # if idx_traj not in [0, 266]:
        #     continue
        print("Processing %sth trajectory of %s" % (idx_traj, cfg.trajectory_path))
        
        dir_path = '/'.join(cfg.trajectory_path.split('/')[:-1])
        meta_infos_found = False
        if os.path.exists(os.path.join(dir_path, 'nir_visible_%s_meta_infos.pkl' % cfg.split)):
            with open(os.path.join(dir_path, 'nir_visible_%s_meta_infos.pkl' % cfg.split), 'rb') as f:
                meta_infos_val = pkl.load(f)
            print(" *Using the predefined ISP parameters in %s" % os.path.join(dir_path, 'nir_visible_%s_meta_infos.pkl' % cfg.split))
            meta_infos_found = True
            image_processing_params = {'random_ccm': cfg.random_ccm, 'random_gains': cfg.random_gains, 'smoothstep': cfg.smoothstep, 'gamma': cfg.gamma, 'add_noise': cfg.add_noise,
                                       'predefined_params': meta_infos_val}

        else:
            print(" *Using random ISP parameters")
            meta_infos_val = {} 
            image_processing_params = {'random_ccm': cfg.random_ccm, 'random_gains': cfg.random_gains, 'smoothstep': cfg.smoothstep, 'gamma': cfg.gamma, 'add_noise': cfg.add_noise}
            
        # transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())
        transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True))

        if cfg.burst_need_downsample == 'true':
            cfg.burst_need_downsample = True
        else:
            cfg.burst_need_downsample = False

        data_processing_val = processing.VisibleBurstProcessing((cfg.crop_sz[0], cfg.crop_sz[1]),
                                                                cfg.downsample_factor,
                                                                transform=transform_val,
                                                                image_processing_params=image_processing_params,
                                                                random_crop=False,
                                                                random_flip=False,
                                                                return_rgb_busrt=cfg.return_rgb_burst,
                                                                burst_need_downsample=cfg.burst_need_downsample)        
        dataset_val = sampler.IndexedImage(nir_visible_val, processing=data_processing_val)
        
        process_fn = SimplePostProcess(return_np=True)

        """The fourth part is to perform prediction"""
        for idx, data in enumerate(dataset_val):

            burst = data['burst']
            gt = data['frame_gt']
            if meta_infos_found:
                meta_info = meta_infos_val['%s' % data['image_name']]
            else:
                meta_info = data['meta_info']
                meta_infos_val['%s' % data['image_name']] = meta_info
                with open(os.path.join(dir_path, 'nir_visible_%s_meta_infos.pkl' % cfg.split), 'wb') as f:
                    pkl.dump(meta_infos_val, f)
                    
            if int(cfg.specify_image_id) != -1:
                if idx != int(cfg.specify_image_id):
                    print("current idx is: ", idx)
                    continue
                
            burst_rgb = data['burst_rgb']
            assert cfg.return_rgb_burst, "Better open this button to save the results."
            meta_info['frame_num'] = idx
            burst_name = '%s' % data['image_name']

            burst = burst.to(device).unsqueeze(0)
            gt = gt.to(device)

            if n.burst_sz is not None:
                burst = burst[:, :n.burst_sz]

            if using_saved_results:
                net_pred = cv2.imread('{}/{}_SR.png'.format(cfg.use_saved_results, burst_name), cv2.IMREAD_UNCHANGED)
                net_pred = (torch.from_numpy(net_pred.astype(np.float32)) / 2 ** 14).permute(2, 0, 1).float().to(device)
                net_pred = net_pred.unsqueeze(0)
            else:
                with torch.no_grad():
                    net_pred, _ = net(burst)
                
                # print("net_pred size: ", net_pred.size())
                # Perform quantization to be consistent with evaluating on saved images
                # net_pred_int = (net_pred.clamp(0.0, 1.0) * 2 ** 14).short()
                # net_pred = net_pred_int.float() / (2 ** 14)

            for m, m_fn in metrics_all.items():
                if 'lr' in m:
                    continue
                metric_value = m_fn(net_pred, gt.unsqueeze(0)).cpu().item()
                scores[m].append(metric_value)

            # Here we want to save result for visualization
            if cfg.save_results and cfg.save_path is not None:
                if idx in selected_images_id or int(cfg.specify_image_id) != -1:
                    if not os.path.isdir(cfg.save_path):
                        os.makedirs('{}'.format(cfg.save_path), exist_ok=True)
                        
                    save_path_traj = os.path.join(cfg.save_path, '{:04d}'.format(idx_traj))
                    if not os.path.isdir(save_path_traj):
                        os.makedirs(save_path_traj, exist_ok=True)

                    # net_pred_np = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(
                    #     np.uint16)
                    HR_image = process_fn.process(gt.cpu(), meta_info)
                    LR_image = process_fn.process(burst_rgb[0], meta_info)
                    SR_image = process_fn.process(net_pred.squeeze(0).cpu(), meta_info)
                    
                    # HR_image = gt.cpu()
                    # LR_image = burst_rgb[0]
                    # SR_image = net_pred.squeeze(0).cpu()
                    
                    # HR_image = (HR_image.permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).numpy().astype(np.uint16)
                    # LR_image = (LR_image.permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).numpy().astype(np.uint16)
                    # SR_image = (SR_image.permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).numpy().astype(np.uint16)
                    
                    # HR_image = cv2.resize(HR_image, dsize=(gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # LR_image = cv2.resize(LR_image, dsize=(HR_image.shape[1], HR_image.shape[0]), interpolation=cv2.INTER_CUBIC)
                    LR_image_upsized = cv2.resize(LR_image, dsize=(HR_image.shape[1], HR_image.shape[0]), interpolation=cv2.INTER_CUBIC)
                    # SR_image = cv2.resize(SR_image, dsize=(gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # HR_image_cvwrite = HR_image[:, :, [2, 1, 0]]
                    # LR_image_cvwrite = LR_image[:, :, [2, 1, 0]]
                    # SR_image_cvwrite = SR_image[:, :, [2, 1, 0]]
                    
                    burst_rgb_np = burst_rgb[0].permute(1, 2, 0).numpy()
                    burst_rgb_np = cv2.resize(burst_rgb_np, dsize=(HR_image.shape[1], HR_image.shape[0]), interpolation=cv2.INTER_CUBIC)
                    burst_rgb_tensor = torch.from_numpy(burst_rgb_np)
                    burst_rgb_tensor = burst_rgb_tensor.permute(2,0,1).to(device)
                    cv2.imwrite('{}/{}_HR.png'.format(save_path_traj, burst_name.split('.')[0]), HR_image)
                    cv2.imwrite('{}/{}_LR_upsized.png'.format(save_path_traj, burst_name.split('.')[0]), LR_image_upsized)
                    cv2.imwrite('{}/{}_LR.png'.format(save_path_traj, burst_name.split('.')[0]), LR_image)                    
                    cv2.imwrite('{}/{}_SR.png'.format(save_path_traj, burst_name.split('.')[0]), SR_image)
                    try:
                        scores['lr_psnr'].append(metrics_all['lr_psnr'](burst_rgb_tensor.unsqueeze(0), gt.unsqueeze(0)).cpu().item())
                        scores['lr_ssim'].append(metrics_all['lr_ssim'](burst_rgb_tensor.unsqueeze(0), gt.unsqueeze(0)).cpu().item())
                    except:
                        print('(warining!) Evaluated %s/%s images of %s/%s, it problemed lrpsnr is %s, ssim is %s' % (idx, len(dataset_val)-1, cfg.dataset_path, burst_name, metrics_all['psnr'](burst_rgb_tensor.unsqueeze(0), gt.unsqueeze(0)), metrics_all['ssim'](burst_rgb_tensor.unsqueeze(0), gt.unsqueeze(0))))
                    print(" Evaluated %s/%s images of %s/%s, its psnr is %s, its ssim is %s, LRPSNR is %s, LRSSIM is %s" % (idx, len(dataset_val)-1, cfg.dataset_path, burst_name, scores['psnr'][-1], scores['ssim'][-1], metrics_all['psnr'](burst_rgb_tensor.unsqueeze(0), gt.unsqueeze(0)), metrics_all['ssim'](burst_rgb_tensor.unsqueeze(0), gt.unsqueeze(0))))
                else:
                    print(" Evaluated %s/%s images of %s/%s, its psnr is %s, its ssim is %s" % (idx, len(dataset_val)-1, cfg.dataset_path, burst_name, scores['psnr'][-1], scores['ssim'][-1]))
            else:
                print(" Evaluated %s/%s images of %s/%s, its psnr is %s, its ssim is %s" % (idx, len(dataset_val)-1, cfg.dataset_path, burst_name, scores['psnr'][-1], scores['ssim'][-1]))

            if int(cfg.specify_image_id) != -1:
                break
   
        # scores_all[n.get_display_name()] = scores
        scores_all_mean['%s_%sth-Traj' % (cfg.ckpt_path.split('/')[-2], idx_traj)] = {m: sum(s) / len(s) for m, s in scores.items()}
        if not os.path.isdir(cfg.save_path):
            os.makedirs('{}'.format(cfg.save_path), exist_ok=True)
        with open(os.path.join(cfg.save_path, 'results_of_%s-%s.pkl' % (cfg.ckpt_path.split('/')[-2], cfg.trajectory_path.split('/')[-1].split('.')[0])), 'wb') as f:
            pkl.dump(scores_all_mean, f)


if __name__ == '__main__':
    main()
