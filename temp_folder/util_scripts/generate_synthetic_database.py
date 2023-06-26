import dataset as datasets
from data import processing, sampler
import data.transforms as tfm
import os
import numpy as np
import cv2
import pickle as pkl


def main():
    image_folder_path = '/home/zheng/projects/deep-burst-sr/downloaded_datasets/Database_for_optimal-SR/NightCity_1024x512/val'
    out_dir = 'downloaded_datasets/Database_for_optimal-SR/NightCity_1024x512/burst4_00011110'
    
    return_rgb_busrt = True

    crop_sz = (1024 + 24*2, 1024 + 24*2)
    burst_sz = 4 #TODO: need to specify if want to generate database
    downsample_factor = 2

    burst_transformation_params = {'max_translation': 24.0,
                                   'max_rotation': 1.0,
                                   'max_shear': 0.0,
                                   'max_scale': 0.0,
                                   'border_crop': 24,
                                   'random_pixelshift': False,
                                   'specified_translation': np.array[[0,0],
                                                                     [0,1],
                                                                     [1,1],
                                                                     [1,0]]} # TODO: need to specify if want to generate database
    image_processing_params = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True,
                               'add_noise': True, 'noise_type': 'unprocessing'}

    image_dataset = datasets.ImageFolder(root=image_folder_path)

    transform_list = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())
    # data_processing = processing.SyntheticBurstProcessing(crop_sz, burst_sz, downsample_factor,
    #                                                       burst_transformation_params=burst_transformation_params,
    #                                                       transform=transform_list,
    #                                                       image_processing_params=image_processing_params)
    data_processing = processing.SyntheticBurstDatabaseProcessing(crop_sz, burst_sz, downsample_factor,
                                                          burst_transformation_params=burst_transformation_params,
                                                          transform=transform_list,
                                                          image_processing_params=image_processing_params,
                                                          return_rgb_busrt=return_rgb_busrt) # TODO: you can set this false if want to save some space
    
    
    
    dataset = sampler.IndexedImage(image_dataset, processing=data_processing)

    for i, d in enumerate(dataset):
        burst = d['burst']
        gt = d['frame_gt']
        meta_info = d['meta_info']
        meta_info['frame_num'] = i

        os.makedirs('{}/{:04d}'.format(out_dir, i), exist_ok=True)
        
        if return_rgb_busrt:
            burst_rgb = d['burst_rgb']
            os.makedirs('{}/{:04d}/burst_rgb'.format(out_dir, i), exist_ok=True)

        burst_np = (burst.permute(0, 2, 3, 1).clamp(0.0, 1.0) * 2**14).numpy().astype(np.uint16)

        for bi, b in enumerate(burst_np):
            cv2.imwrite('{}/{:04d}/im_raw_{:02d}.png'.format(out_dir, i, bi), b)
            burst_rgb_np = (burst_rgb[bi].permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).numpy().astype(np.uint16)
            cv2.imwrite('{}/{:04d}/burst_rgb/{:02d}.png'.format(out_dir, i, bi), burst_rgb_np)

        gt_np = (gt.permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).numpy().astype(np.uint16)
        cv2.imwrite('{}/{:04d}/im_rgb.png'.format(out_dir, i), gt_np)

        with open('{}/{:04d}/meta_info.pkl'.format(out_dir, i), "wb") as file_:
            pkl.dump(meta_info, file_, -1)


if __name__ == '__main__':
    main()

