import dataset as datasets
from data import processing, sampler
import data.transforms as tfm
import os
import numpy as np
import cv2
import pickle as pkl
import random
import subprocess



def main():
    image_folder_path = '/mnt/samsung/zheng/downloaded_datasets/NightCity_1024x512/val'
    img_names = os.listdir(image_folder_path)
    for img_name in img_names:
        val_data_path = os.path.join(image_folder_path, '..', 'val_' + img_name[:-4].split('_')[0] + '-' + img_name[:-4].split('_')[1])
        os.makedirs(val_data_path, exist_ok=True)
        # 运行copy new val dir程序
        cmd = f'cp -r {image_folder_path}/{img_name} {val_data_path}/.'
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            print(f"Error: Failed to copy new val dir {val_data_path}")
            continue
        
        # define the number of steps
        for step in np.arange(4,17,4):
            output_step_folder_path = os.path.join(image_folder_path, '..', 'step_%s_%s_small-shift' % (step, img_name[:-4]))
            os.makedirs(output_step_folder_path, exist_ok=True)
            f=open(os.path.join(output_step_folder_path, 'step%s_database.txt' % step), 'a')
            permutations = []
            while len(permutations)!=100:
                p = np.random.randint(-3,4,size=(step,2))
                p[0,0]=0
                p[0,1]=0
                a=p[1:, :]
                a=a[np.argsort(a[:, 0])]
                p[1:, :] = a
                found = False
                for j in permutations:
                    if np.allclose(j,p):
                        found = True
                        break
                if found:
                    continue
                else:
                    permutations.append(p)
                    for num in p.flatten():
                        f.write(str(num)+' ')
                    f.write('\n')
                    # print("%s" % p.flatten(), file=f)
            f.close()
            f_step_result_file = open(os.path.join(output_step_folder_path, "result.txt"), 'a')
            # define the trajectory of pixel shift
            for idx, permutation in enumerate(permutations):       
                out_dir = os.path.join(output_step_folder_path, "%sth_trajectory" % idx)
                os.makedirs(out_dir, exist_ok=True)
                return_rgb_busrt = False

                # crop_sz = (1024 + 24*2, 1024 + 24*2)
                crop_sz = (384, 384) # (384, 384)
                burst_sz = step #TODO: need to specify if want to generate database
                downsample_factor = 4

                burst_transformation_params = {'max_translation': 24.0,
                                            'max_rotation': 1.0,
                                            'max_shear': 0.0,
                                            'max_scale': 0.0,
                                            'border_crop': 24,
                                            'random_pixelshift': False,
                                            'specified_translation': permutation} # TODO: need to specify if want to generate database
                image_processing_params = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True,
                                        'add_noise': True, 'noise_type': 'unprocessing'}
                # image_processing_params = {'random_ccm': False, 'random_gains': False, 'smoothstep': False, 'gamma': False,
                #                         'add_noise': False, 'noise_type': 'unprocessing'}

                image_dataset = datasets.ImageFolder(root=val_data_path)

                transform_list = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())
                # data_processing = processing.SyntheticBurstProcessing(crop_sz, burst_sz, downsample_factor,
                #                                                       burst_transformation_params=burst_transformation_params,
                #                                                       transform=transform_list,
                #                                                       image_processing_params=image_processing_params)
                data_processing = processing.SyntheticBurstDatabaseProcessing(crop_sz, burst_sz, downsample_factor,
                                                                    burst_transformation_params=burst_transformation_params,
                                                                    transform=transform_list,
                                                                    image_processing_params=image_processing_params,
                                                                    return_rgb_busrt=return_rgb_busrt,
                                                                    random_crop=False) # TODO: you can set this false if want to save some space
                
                
                
                dataset = sampler.IndexedImage(image_dataset, processing=data_processing)

                os.makedirs('{}/bursts'.format(out_dir), exist_ok=True)
                os.makedirs('{}/gt'.format(out_dir), exist_ok=True)

                for i, d in enumerate(dataset):
                    burst = d['burst']
                    gt = d['frame_gt']
                    meta_info = d['meta_info']
                    meta_info['frame_num'] = i

                    
                    os.makedirs('{}/bursts/{:04d}'.format(out_dir, i), exist_ok=True)
                    os.makedirs('{}/gt/{:04d}'.format(out_dir, i), exist_ok=True)
                    
                    if return_rgb_busrt:
                        burst_rgb = d['burst_rgb']
                        os.makedirs('{}/{:04d}/burst_rgb'.format(out_dir, i), exist_ok=True)

                    burst_np = (burst.permute(0, 2, 3, 1).clamp(0.0, 1.0) * 2**14).numpy().astype(np.uint16)

                    for bi, b in enumerate(burst_np):
                        cv2.imwrite('{}/bursts/{:04d}/im_raw_{:02d}.png'.format(out_dir, i, bi), b)
                        if return_rgb_busrt:
                            burst_rgb_np = (burst_rgb[bi].permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).numpy().astype(np.uint16)
                            cv2.imwrite('{}/{:04d}/burst_rgb/{:02d}.png'.format(out_dir, i, bi), burst_rgb_np)

                    gt_np = (gt.permute(1, 2, 0).clamp(0.0, 1.0) * 2 ** 14).numpy().astype(np.uint16)
                    cv2.imwrite('{}/gt/{:04d}/im_rgb.png'.format(out_dir, i), gt_np)

                    with open('{}/gt/{:04d}/meta_info.pkl'.format(out_dir, i), "wb") as file_:
                        pkl.dump(meta_info, file_, -1)

                print("---------------------------------------------------------!!!!!!!!!%s is performing!!!!!!!!!" % idx, file=f_step_result_file)
                
                # 运行evaluation程序
                cmd = f'python evaluation/synburst/compute_score.py dbsr_default --burst_size {step} --dataset_root {out_dir}'
                try:
                    subprocess.run(cmd, shell=True, check=True, stdout=f_step_result_file)
                except subprocess.CalledProcessError:
                    print(f"Error: Failed to run dataset {out_dir}", file=f_step_result_file)
                    continue
                

if __name__ == '__main__':
    main()

