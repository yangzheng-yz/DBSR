# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from dataset.base_image_dataset import BaseImageDataset
from data.image_loader import opencv_loader
from admin.environment import env_settings

class nir_visible(BaseImageDataset):
    """
    Dataset class for loading the Canon RGB images from the Zurich RAW 2 RGB dataset [1]

    [1] Replacing Mobile Camera ISP with a Single Deep Learning Model. Andrey Ignatov, Luc Van Gool and Radu Timofte,
        CVPRW 2020
    """
    def __init__(self, root=None, split='train', image_loader=opencv_loader, initialize=True, burst_sz=16):
        """
        args:
            root - Path to root dataset directory
            split - Dataset split to use. Can be 'train' or 'test'
            image_loader - loader used to read the images
            initialize - boolean indicating whether to load the meta-data for the dataset
        """
        root = env_settings().nir_visible_dir if root is None else root
        super().__init__('nir_visible', root, image_loader)
        self.split = split
        self.burst_sz = burst_sz

        if initialize:
            self.initialize()

    def initialize(self):
        split = self.split
        root = self.root
        if split in ['train', 'test', 'train-1', 'test-1', 'train-2']:
            self.img_pth = os.path.join(root, split)
        else:
            raise Exception('Unknown split {}'.format(split))

        self.image_list = self._get_image_list(split)

    def _get_image_list(self, split):
        image_list = [folder for folder in os.listdir(self.img_pth) if os.path.isdir(os.path.join(self.img_pth, folder))]
        image_list.sort()
        return image_list

    def _get_image(self, im_id, sub_dir):
        path = os.path.join(self.img_pth, self.image_list[im_id], sub_dir)
        img_files = sorted(os.listdir(path))
        img_list = []
        for img_file in img_files:
            img_path = os.path.join(path, img_file)
            img = self.image_loader(img_path)
            if len(img.shape) == 2:
                # Convert grayscale image to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if sub_dir == "burst" and len(img_list) > self.burst_sz:
                break
            img_list.append(img)
        return img_list

    def get_image_info(self, im_id):
        return self.image_list[im_id]

    def get_image(self, im_id, info=None):
        gt_img = self._get_image(im_id, "gt")[0]
        burst_imgs = self._get_image(im_id, "burst")

        if info is None:
            info = self.get_image_info(im_id)

        frame = {
            'gt': gt_img,
            'burst': burst_imgs,
        }

        return frame, info