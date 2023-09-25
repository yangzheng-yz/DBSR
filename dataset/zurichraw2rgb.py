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
import pickle as pkl

class ZurichRAW2RGB(BaseImageDataset):
    """
    Dataset class for loading the Canon RGB images from the Zurich RAW 2 RGB dataset [1]

    [1] Replacing Mobile Camera ISP with a Single Deep Learning Model. Andrey Ignatov, Luc Van Gool and Radu Timofte,
        CVPRW 2020
    """
    def __init__(self, root=None, split='train', image_loader=opencv_loader, initialize=True):
        """
        args:
            root - Path to root dataset directory
            split - Dataset split to use. Can be 'train' or 'test'
            image_loader - loader used to read the images
            initialize - boolean indicating whether to load the meta-data for the dataset
        """
        root = env_settings().zurichraw2rgb_dir if root is None else root
        super().__init__('ZurichRAW2RGB', root, image_loader)
        self.split = split

        if initialize:
            self.initialize()

    def initialize(self):
        split = self.split
        root = self.root
        if split in ['train', 'test']:
            self.img_pth = os.path.join(root, split, 'canon')
        elif split in ['val']:
            self.img_pth = os.path.join(root, split, 'gt')
        else:
            raise Exception('Unknown split {}'.format(split))

        self.image_list = self._get_image_list(split)

    def _get_image_list(self, split):
        if split == 'train':
            image_list = ['{:d}.jpg'.format(i) for i in range(46839)]
        elif split == 'test':
            image_list = ['{:d}.jpg'.format(i) for i in range(1204)]
        elif split == 'val':
            image_list = ['%.4d' % i for i in range(300)] 
        else:
            raise Exception

        return image_list

    def get_image_info(self, im_id):
        return "%s_%s" % (self.split, im_id)

    def _get_image(self, im_id):
        path = os.path.join(self.img_pth, self.image_list[im_id])
        img = self.image_loader(path)
        return img
    
    def _get_val_image(self, im_id):
        img_path = os.path.join(self.img_pth, self.image_list[im_id], 'im_rgb.png')
        info_path = os.path.join(self.img_pth, self.image_list[im_id], 'meta_info.pkl')
        f=open(info_path, 'rb')
        img = self.image_loader(img_path)
        meta_info = pkl.load(f)
        f.close()
        return img, meta_info

    def get_image(self, im_id, info=None):
        if self.split in ['train', 'test']:
            frame = self._get_image(im_id)

            if info is None:
                info = self.get_image_info(im_id)
        elif self.split in ['val']:
            frame, info = self._get_val_image(im_id)
            print("type of frame: ", type(frame))
        else:
            Exception

        return frame, info
