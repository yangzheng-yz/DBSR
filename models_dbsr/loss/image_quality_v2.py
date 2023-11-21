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

import torch
import torch.nn as nn
import torch.nn.functional as F
import models_dbsr.loss.msssim as msssim
import models_dbsr.loss.spatial_color_alignment as sca_utils
import math
import lpips
from torchvision.models import vgg19

def contrast_loss(pred, gt):
    pred_contrast = pred.std(dim=(2, 3)) - pred.mean(dim=(2, 3))
    gt_contrast = gt.std(dim=(2, 3)) - gt.mean(dim=(2, 3))
    return F.l1_loss(pred_contrast, gt_contrast)

class BackgroundLoss(nn.Module):
    def __init__(self, background_threshold=0.1):
        super().__init__()
        self.background_threshold = background_threshold

    def forward(self, prediction, target):
        # Create a mask where the target is below the background threshold
        background_mask = (target < self.background_threshold).float()
        
        # Calculate the loss as the mean of the product of prediction and the background mask
        loss = (prediction * background_mask).mean()
        return loss

class ContrastLoss(nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()

    def forward(self, input, target):
        # Calculate the contrast of the input and target
        input_contrast = self._contrast(input)
        target_contrast = self._contrast(target)

        # Calculate the noise level in the background
        input_noise = self._noise(input)
        target_noise = self._noise(target)

        # Calculate loss based on contrast difference and noise level
        contrast_loss = F.l1_loss(input_contrast, target_contrast)
        noise_loss = F.l1_loss(input_noise, target_noise)

        # Combine the losses
        total_loss = contrast_loss + noise_loss
        return total_loss

    def _contrast(self, x):
        # Kernel for edge detection, e.g., Sobel operator
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).to(x.device)
        kernel = kernel.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
        contrast = F.conv2d(x, kernel, padding=1, groups=x.size(1))
        return contrast

    def _noise(self, x):
        # Simple standard deviation to estimate noise
        noise = torch.std(x, dim=(2, 3), keepdim=True)
        return noise
    
class GANLoss(nn.Module):
    def __init__(self, discriminator, real_label=1.0, fake_label=0.0, device='cuda'):
        super(GANLoss, self).__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.loss = nn.BCEWithLogitsLoss()
        self.discriminator = discriminator.to(device)
        self.device = device

    def get_discriminator_output(self, images):
        return self.discriminator(images).view(-1)

    def forward(self, fake_images, real_images=None, target_is_real=True):
        if real_images is not None:
            # When real images are provided, use them to compute the discriminator's output
            discriminator_output = self.get_discriminator_output(real_images)
        else:
            # Otherwise, use the fake images
            discriminator_output = self.get_discriminator_output(fake_images)

        if target_is_real:
            # If the target is real, the labels should be all ones
            labels = torch.full((discriminator_output.size(0),), self.real_label, device=self.device)
        else:
            # If the target is fake, the labels should be all zeros
            labels = torch.full((discriminator_output.size(0),), self.fake_label, device=self.device)

        loss = self.loss(discriminator_output, labels)
        return loss

class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):  # Add a device parameter
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:18]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_extractor = self.feature_extractor.to(device)  # Move the model to the specified device

    def forward(self, input, target):
        input_features = self.feature_extractor(input)
        target_features = self.feature_extractor(target)
        loss = nn.functional.mse_loss(input_features, target_features)
        return loss


class PixelWiseError(nn.Module):
    """ Computes pixel-wise error using the specified metric. Optionally boundary pixels are ignored during error
        calculation """
    def __init__(self, metric='l1', boundary_ignore=None):
        super().__init__()
        self.boundary_ignore = boundary_ignore

        if metric == 'l1':
            self.loss_fn = F.l1_loss
        elif metric == 'l2':
            self.loss_fn = F.mse_loss
        elif metric == 'l2_sqrt':
            def l2_sqrt(pred, gt):
                return (((pred - gt) ** 2).sum(dim=-3)).sqrt().mean()
            self.loss_fn = l2_sqrt
        elif metric == 'charbonnier':
            def charbonnier(pred, gt):
                eps = 1e-3
                return ((pred - gt) ** 2 + eps**2).sqrt().mean()
            self.loss_fn = charbonnier
        elif metric == 'perceptual':
            self.loss_fn = PerceptualLoss()
        else:
            raise Exception

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            # Remove boundary pixels
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Valid indicates image regions which should be used for loss calculation
        if valid is None:
            # print("pred: ", pred.size())
            # print("gt: ", gt.size())
            # Check if sizes of pred and gt are different
            if pred.size() != gt.size():
                # Resize gt to match the size of pred
                gt = F.interpolate(gt, size=(pred.size(2), pred.size(3)), mode='bilinear', align_corners=False)
            else:
                gt = gt
            err = self.loss_fn(pred, gt)
        else:
            err = self.loss_fn(pred, gt, reduction='none')

            eps = 1e-12
            elem_ratio = err.numel() / valid.numel()
            err = (err * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)

        return err


class PSNR(nn.Module):
    def __init__(self, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = PixelWiseError(metric='l2', boundary_ignore=boundary_ignore)
        self.max_value = max_value

    def psnr(self, pred, gt, valid=None):
        mse = self.l2(pred, gt, valid=valid)

        if getattr(self, 'max_value', 1.0) is not None:
            psnr = 20 * math.log10(getattr(self, 'max_value', 1.0)) - 10.0 * mse.log10()
        else:
            psnr = 20 * gt.max().log10() - 10.0 * mse.log10()

        if torch.isinf(psnr) or torch.isnan(psnr):
            print('invalid psnr')

        return psnr

    def forward(self, pred, gt, valid=None, batch=False):
        if valid is None:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0)) for p, g in
                        zip(pred, gt)]
        else:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), v.unsqueeze(0)) for p, g, v in zip(pred, gt, valid)]

        psnr_all = [p for p in psnr_all if not (torch.isinf(p) or torch.isnan(p))]

        if len(psnr_all) == 0:
            psnr = 0
        else:
            psnr = sum(psnr_all) / len(psnr_all)
        
        if batch:
            return psnr_all
        else:
            return psnr


class SSIM(nn.Module):
    def __init__(self, boundary_ignore=None, use_for_loss=True):
        super().__init__()
        self.ssim = msssim.SSIM(spatial_out=True)
        self.boundary_ignore = boundary_ignore
        self.use_for_loss = use_for_loss

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)

        loss = self.ssim(pred, gt)

        if valid is not None:
            valid = valid[..., 5:-5, 5:-5]  # assume window size 11

            eps = 1e-12
            elem_ratio = loss.numel() / valid.numel()
            loss = (loss * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)
        else:
            loss = loss.mean()

        if self.use_for_loss:
            loss = 1.0 - loss
        return loss


class LPIPS(nn.Module):
    def __init__(self, boundary_ignore=None, type='alex', bgr2rgb=False):
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.bgr2rgb = bgr2rgb

        if type == 'alex':
            self.loss = lpips.LPIPS(net='alex')
        elif type == 'vgg':
            self.loss = lpips.LPIPS(net='vgg')
        else:
            raise Exception

    def forward(self, pred, gt, valid=None):
        if self.bgr2rgb:
            pred = pred[..., [2, 1, 0], :, :].contiguous()
            gt = gt[..., [2, 1, 0], :, :].contiguous()

        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        loss = self.loss(pred, gt)

        return loss.mean()


class AlignedL2(nn.Module):
    """ Computes L2 error after performing spatial and color alignment of the input image to GT"""
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sca = sca_utils.SpatialColorAlignment(alignment_net, sr_factor)
        self.boundary_ignore = boundary_ignore

    def forward(self, pred, gt, burst_input):
        pred_warped_m, valid = self.sca(pred, gt, burst_input)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Estimate MSE
        mse = F.mse_loss(pred_warped_m, gt, reduction='none')

        eps = 1e-12
        elem_ratio = mse.numel() / valid.numel()
        mse = (mse * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return mse
