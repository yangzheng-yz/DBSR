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

class MixedMiceNIR_Dai(BaseImageDataset):

    def __init__(self, root=None, split='train', image_loader=opencv_loader, initialize=True):
        """
        args:
            root - Path to root dataset directory
            split - Dataset split to use. Can be 'train' or 'test'
            image_loader - loader used to read the images
            initialize - boolean indicating whether to load the meta-data for the dataset
        """
        root = env_settings().MixedMiceNIR_Dai_dir if root is None else root
        super().__init__('MixedMiceNIR_Dai', root, image_loader)
        self.split = split

        if initialize:
            self.initialize()

    def initialize(self):
        split = self.split
        root = self.root
        if split in ['train', 'val', 'test', 'val_two']:
            self.img_pth = os.path.join(root, split, 'without_art')
        elif split == 'real_test_one':
            self.img_pth = os.path.join(root, split)
        else:
            raise Exception('Unknown split {}'.format(split))

        self.image_list = self._get_image_list(split)

    def _get_image_list(self, split):
        if split == 'train':
            image_list = os.listdir(self.img_pth)
            image_list.sort()
        elif split == 'test':
            image_list = os.listdir(self.img_pth)
            image_list.sort()
        elif split == 'val' or 'val_two':
            image_list = os.listdir(self.img_pth)
            image_list.sort() 
        elif split == 'real_test_one':
            image_list = [os.listdir(self.img_pth)]
            image_list.sort()
        else:
            raise Exception

        return image_list

    def get_image_info(self, im_id):
        return "%s_%s" % (self.split, self.image_list[im_id].split('.png')[0])

    def _get_image(self, im_id):
        path = os.path.join(self.img_pth, self.image_list[im_id])
        img = self.image_loader(path)
        return img

    def get_image(self, im_id, info=None):
        if self.split in ['train', 'test', 'val', 'val_two']:
            frame = self._get_image(im_id)

            if info is None:
                info = self.get_image_info(im_id)
        elif self.split == 'real_test_one':
            frames = []
            for idx, f in enumerate(self.image_list):
                img = (self._get_image(idx) / 65535.0 * 255.0).astype(np.uint8)
                 
                frames.append(img[0:512, 0:512])
            return frames, None
        else:
            Exception

        return frame, info
