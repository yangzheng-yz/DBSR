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

from actors.base_actor import BaseActor
from models.loss.spatial_color_alignment import SpatialColorAlignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, num_frames, num_channels, hidden_size):
        super(ActorCritic, self).__init__()
        
        # Actor Network
        self.actor_conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.actor_lstm = nn.LSTM(147456, hidden_size, batch_first=True)
        self.actor_linear = nn.Linear(hidden_size, 5 * (num_frames - 1))
        
        # Critic Network
        self.critic_conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.critic_linear = nn.Linear(147456, 1)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)  # Reshape to [batch_size*num_frames, channels, height, width]
        
        # Actor
        x_actor = self.actor_conv(x)
        x_actor = x_actor.view(batch_size, num_frames, -1)  # Reshape back to [batch_size, num_frames, features]
        _, (h_n, _) = self.actor_lstm(x_actor)
        action_logits = self.actor_linear(h_n.squeeze(0))
        action_logits = action_logits.view(batch_size, num_frames - 1, 5)  # Reshape to [batch_size, num_frames-1, 5]
        probs = F.softmax(action_logits, dim=-1)
        dists = [Categorical(p) for p in probs.split(1, dim=1)]
        
        # Critic
        x_critic = self.critic_conv(x)
        x_critic = x_critic.view(batch_size, num_frames, -1).mean(dim=1)  # Average over frames
        value = self.critic_linear(x_critic)
        
        return dists, value

class DBSR_PSNetActor(BaseActor):
    """Actor for training Pixel Shift reinforcement learning model on synthetic bursts """
    def __init__(self, sr_encoder, sr_merging, net, objective, loss_weight=None):
        super().__init__(net, objective)
        self.sr_encoder = sr_encoder
        self.sr_merging = sr_merging

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        print("!!!!!!!!!!!!!!!!policynet and sr_net's device: ", device)
        self.net.to(device)
        self.sr_encoder.to(device)
        self.sr_merging.to(device)

    def __call__(self, data):
        # Run DBSR encoder
        with torch.no_grad():
            encoded_burst = self.sr_encoder(data)
            merged_feature = self.sr_merging(encoded_burst)

        # Run policy network
        actions_logits = self.net(merged_feature['fused_enc'])

        # Compute action probabilities
        actions_pdf = torch.nn.functional.softmax(actions_logits, dim=-1) # []

        return actions_pdf



class DBSRSyntheticActor(BaseActor):
    """Actor for training DBSR model on synthetic bursts """
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'rgb': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        # Run network
        # print("net's device: ", next(self.net.parameters()).device)
        # print("data burst info: ", type(data['burst']))
        # print(data['burst'].size())
        pred, aux_dict = self.net(data['burst'])

        # Compute loss
        loss_rgb_raw = self.objective['rgb'](pred, data['frame_gt'])
        loss_rgb = self.loss_weight['rgb'] * loss_rgb_raw
        
        if self.objective.get('perceptual', None) is not None:
            loss_percept_raw = self.objective['perceptual'](pred, data['frame_gt'])
            loss_percept = self.loss_weight['perceptual'] * loss_percept_raw
            loss_rgb += loss_percept

        if 'psnr' in self.objective.keys():
            psnr = self.objective['psnr'](pred.clone().detach(), data['frame_gt'])

        loss = loss_rgb
        if self.objective.get('perceptual', None) is not None:
            stats = {'Loss/total': loss.item(),
                    'Loss/rgb': loss_rgb.item() - loss_percept.item(),
                    'Loss/percept': loss_percept.item(),
                    'Loss/raw/rgb': loss_rgb_raw.item()}
        else: 
            stats = {'Loss/total': loss.item(),
                    'Loss/rgb': loss_rgb.item(),
                    'Loss/raw/rgb': loss_rgb_raw.item()}

        if 'psnr' in self.objective.keys():
            stats['Stat/psnr'] = psnr.item()

        return loss, stats


class DBSRRealWorldActor(BaseActor):
    """Actor for training DBSR model on real-world bursts from BurstSR dataset"""
    def __init__(self, net, objective, alignment_net, loss_weight=None, sr_factor=4):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'rgb': 1.0}

        self.sca = SpatialColorAlignment(alignment_net.eval(), sr_factor=sr_factor)
        self.loss_weight = loss_weight

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        print("!!!!!!!!!!!!!!!!net and sca's device: ", device)
        self.net.to(device)
        self.sca.to(device)

    def __call__(self, data):
        # Run network
        gt = data['frame_gt']
        burst = data['burst']
        pred, aux_dict = self.net(burst)

        # Perform spatial and color alignment of the prediction
        pred_warped_m, valid = self.sca(pred, gt, burst)

        # Compute loss
        loss_rgb_raw = self.objective['rgb'](pred_warped_m, gt, valid=valid)

        loss_rgb = self.loss_weight['rgb'] * loss_rgb_raw

        if 'psnr' in self.objective.keys():
            # detach, otherwise there is memory leak
            psnr = self.objective['psnr'](pred_warped_m.clone().detach(), gt, valid=valid)

        loss = loss_rgb

        stats = {'Loss/total': loss.item(),
                 'Loss/rgb': loss_rgb.item(),
                 'Loss/raw/rgb': loss_rgb_raw.item()}

        if 'psnr' in self.objective.keys():
            stats['Stat/psnr'] = psnr.item()

        return loss, stats
