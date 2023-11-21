"""
This version is for the dai's mice dataset.
"""
import utils.data_format_utils as df_utils
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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time

from evaluation.common_utils.network_param import NetworkParam
from models_dbsr.loss.image_quality_v2 import PSNR, SSIM, LPIPS
from data.postprocessing_functions import SimplePostProcess

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
    Zurich_test = datasets.MixedMiceNIR_Dai(root=cfg.dataset_path, split=cfg.split)   
    
    metrics = ('psnr', 'ssim')
    device = 'cuda'
    boundary_ignore = 40
    metrics_all = {}
    scores = {}
    for m in metrics:
        if not cfg.calculate_loss:
            break
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
    assert cfg.ckpt_path is not None, "You must specify a pretrained weights to evaluate."
    # ckpt_root = cfg.ckpt_path[:-25]
    
    # ckpts = [os.path.join(ckpt_root, i) for i in os.listdir(ckpt_root) if int(i.split('_ep')[1].split('.')[0]) > 169]
    # # print(ckpts[0].split('_ep')[1].split('.')[0])
    # # time.sleep(1000)
    # ckpts.sort()
    """The fourth part is to perform prediction"""
    if not os.path.isdir(cfg.save_path):
        os.makedirs('{}'.format(cfg.save_path), exist_ok=True)
    save_txt_path = os.path.join(cfg.save_path, 'metrics_record.txt') # TODO: to recover, uncomment these four lines
    if os.path.exists(save_txt_path):
        os.remove(save_txt_path)
    save_txt = open(save_txt_path, 'a')

    # save_txt_path = os.path.join(cfg.save_path, 'metrics_record_allinone.txt') # TODO: to recover, delete these four lines
    # if os.path.exists(save_txt_path):
    #     os.remove(save_txt_path)
    # save_txt = open(save_txt_path, 'a')
    
# for ckpt_path in ckpts:
    # cfg.ckpt_path = ckpt_path # TODO: to recovery, delete this
    n = NetworkParam(network_path='%s' % cfg.ckpt_path, # both .pth and .pth.tar can be loaded 
                                    unique_name='%s' % cfg.ckpt_path.split('/')[-2])         # Unique name is used when saving results

    net = n.load_net()
    device = 'cuda'
    net.to(device).train(False)
    """The third part is to define the transformation's type"""
    
    scores_all_mean = {}
    selected_images_id = np.arange(0,len(Zurich_test),1) 
        
    # transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())
    transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True))
    
    dir_path = "/home/user/zheng/DBSR/util_scripts"
    
    data_processing_val = processing.RealBurstDatabaseProcessing((640, 512), cfg.burst_sz,
                                                                        cfg.downsample_factor,
                                                                        transform=transform_val,
                                                                        random_crop=False,
                                                                        return_rgb_busrt=cfg.return_rgb_burst)


    dataset_val = sampler.IndexedImage(Zurich_test, processing=data_processing_val)
    
    process_fn = SimplePostProcess(return_np=True)


    
    for idx, (data) in enumerate(dataset_val):

        burst = data['burst']
        gt = data['frame_gt']

        # HR = HR_batch[i].clone().cpu()
        # # print("image type: ", image.device)
        # # time.sleep(1000)
        # burst_transformation_params = {'max_translation': 24.0,
        #                             'max_rotation': 1.0,
        #                             'max_shear': 0.0,
        #                             'max_scale': 0.0,
        #                             'random_pixelshift': False,
        #                             'specified_translation': permutations_batch[i]}
        # image_burst_rgb, _ = syn_burst_generation.single2lrburstdatabase(HR, burst_size=burst_size,
        #                                             downsample_factor=self.downsample_factor,
        #                                             transformation_params=burst_transformation_params,
        #                                             interpolation_type=self.interpolation_type)
        # image_burst = rgb2raw.mosaic(image_burst_rgb.clone())
                
        if int(cfg.specify_image_id) != -1:
            if idx != int(cfg.specify_image_id):
                print("current idx is: ", idx)
                continue
            
        burst_rgb = data['burst_rgb']
        assert cfg.return_rgb_burst, "Better open this button to save the results."
        burst_name = "%s_%s" % (cfg.split, idx)

        burst = burst.to(device).unsqueeze(0)
        gt = gt.to(device)

        # if n.burst_sz is not None:
        #     burst = burst[:, :n.burst_sz]


        with torch.no_grad():
            net_pred, _ = net(burst)
            
            # print("net_pred size: ", net_pred.size())
            # Perform quantization to be consistent with evaluating on saved images
            # net_pred_int = (net_pred.clamp(0.0, 1.0) * 2 ** 14).short()
            # net_pred = net_pred_int.float() / (2 ** 14)

        for m, m_fn in metrics_all.items():
            metric_value = m_fn(net_pred, gt.unsqueeze(0)).cpu().item()
            scores[m].append(metric_value)

        # Here we want to save result for visualization
        if cfg.save_results and cfg.save_path is not None:
            if idx in selected_images_id or int(cfg.specify_image_id) != -1:
                    
                save_path_png = os.path.join(cfg.save_path, 'images')
                if not os.path.isdir(save_path_png):
                    os.makedirs(save_path_png, exist_ok=True)

                # net_pred_np = (net_pred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).cpu().numpy().astype(
                #     np.uint16)
                # print("meta_info: ", meta_info)
                HR_image = process_fn.process(gt.cpu(), data['meta_info'])
                LR_image = process_fn.process(burst_rgb[0], data['meta_info'])
                SR_image = process_fn.process(net_pred.squeeze(0).cpu(), data['meta_info'])
                
                # HR_image = gt.cpu()
                # LR_image = burst_rgb[0]
                # SR_image = net_pred.squeeze(0).cpu()
                
                # HR_image = (HR_image.permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).numpy().astype(np.uint16)
                # LR_image = (LR_image.permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).numpy().astype(np.uint16)
                # SR_image = (SR_image.permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).numpy().astype(np.uint16)
                
                # HR_image = cv2.resize(HR_image, dsize=(gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                LR_image_cubic = cv2.resize(LR_image, dsize=(SR_image.shape[1], SR_image.shape[0]), interpolation=cv2.INTER_CUBIC)
                # SR_image = cv2.resize(SR_image, dsize=(gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                # HR_image_cvwrite = HR_image[:, :, [2, 1, 0]]
                # LR_image_cvwrite = LR_image[:, :, [2, 1, 0]]
                # SR_image_cvwrite = SR_image[:, :, [2, 1, 0]]
                
                burst_rgb_np = burst_rgb[0].permute(1, 2, 0).numpy()
                burst_rgb_np = cv2.resize(burst_rgb_np, dsize=(HR_image.shape[1], HR_image.shape[0]), interpolation=cv2.INTER_CUBIC)
                burst_rgb_tensor = torch.from_numpy(burst_rgb_np)
                burst_rgb_tensor = burst_rgb_tensor.permute(2,0,1).to(device)
                cv2.imwrite('{}/{}_HR.png'.format(save_path_png, burst_name.split('.')[0]), cv2.cvtColor(HR_image, cv2.COLOR_BGR2GRAY))
                cv2.imwrite('{}/{}_LR_cubic.png'.format(save_path_png, burst_name.split('.')[0]), cv2.cvtColor(LR_image_cubic, cv2.COLOR_BGR2GRAY))
                cv2.imwrite('{}/{}_LR.png'.format(save_path_png, burst_name.split('.')[0]), cv2.cvtColor(LR_image, cv2.COLOR_BGR2GRAY))
                cv2.imwrite('{}/{}_SR.png'.format(save_path_png, burst_name.split('.')[0]), cv2.cvtColor(SR_image, cv2.COLOR_BGR2GRAY))

                if not cfg.calculate_loss:
                    print(" Evaluated %s/%s images of %s/%s" % (idx, len(dataset_val)-1, cfg.dataset_path, burst_name), file=save_txt)
                    continue
                print(" Evaluated %s/%s images of %s/%s, its PSNR is %s, its SSIM is %s, LRPSNR is %s, LRSSIM is %s" % (idx, len(dataset_val)-1, cfg.dataset_path, burst_name, scores['psnr'][-1], scores['ssim'][-1], metrics_all['psnr'](burst_rgb_tensor.unsqueeze(0), gt.unsqueeze(0)).cpu().item(), metrics_all['ssim'](burst_rgb_tensor.unsqueeze(0), gt.unsqueeze(0)).cpu().item()), file=save_txt)
            else:
                if not cfg.calculate_loss:
                    print(" Evaluated %s/%s images of %s/%s" % (idx, len(dataset_val)-1, cfg.dataset_path, burst_name), file=save_txt)
                    continue
                print(" Evaluated %s/%s images of %s/%s, its PSNR is %s, its SSIM is %s" % (idx, len(dataset_val)-1, cfg.dataset_path, burst_name, scores['psnr'][-1], scores['ssim'][-1]), file=save_txt)
        else:
            if not cfg.calculate_loss:
                print(" Evaluated %s/%s images of %s/%s" % (idx, len(dataset_val)-1, cfg.dataset_path, burst_name), file=save_txt)
                continue
            # print(" Evaluated %s/%s images of %s/%s, its PSNR is %s, its SSIM is %s" % (idx, len(dataset_val)-1, cfg.dataset_path, burst_name, scores['psnr'][-1], scores['ssim'][-1]), file=save_txt)

        if int(cfg.specify_image_id) != -1:
            break

    # scores_all[n.get_display_name()] = scores
    if cfg.calculate_loss:
        scores_all_mean = {m: sum(s) / len(s) for m, s in scores.items()}
    if not os.path.isdir(cfg.save_path):
        os.makedirs('{}'.format(cfg.save_path), exist_ok=True)
    if cfg.calculate_loss: # TODO: uncomment these three lines
        with open(os.path.join(cfg.save_path, 'results.pkl'), 'wb') as f:
            pkl.dump(scores_all_mean, f)
    # f = print("%s epoch's average psnr is %s, ssim is %s" % (int(cfg.ckpt_path.split('_ep')[1].split('.')[0]), scores_all_mean['psnr'], scores_all_mean['ssim']), file=save_txt) # TODO: to recovery, delete this
    
            


if __name__ == '__main__':
    main()