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
import time
from actors.base_actor import BaseActor
from models_dbsr.loss.spatial_color_alignment import SpatialColorAlignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torchvision.models as models
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, in_channels):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x

class ActorCritic_v3(nn.Module):
    def __init__(self, num_frames, hidden_size):
        super(ActorCritic_v3, self).__init__()
        
        # Shared ResNet Feature Extractor
        self.shared_resnet = ResNet18(in_channels=4)
        
        # Actor Network
        self.actor_lstm = nn.LSTM(512, hidden_size, batch_first=True)
        self.actor_linear = nn.Linear(hidden_size, 5 * (num_frames-1))
        self.num_frames = num_frames
        
        # Critic Network
        self.critic_linear = nn.Linear(512, 1)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        actor_states, critic_states = self._forward_single(x)
        
        # Actor
        _, (h_n, _) = self.actor_lstm(actor_states)
        action_logits = self.actor_linear(h_n.squeeze(0))
        action_logits = action_logits.view(batch_size, self.num_frames - 1, 5)
        
        probs = F.softmax(action_logits, dim=-1)
        dists = [Categorical(p) for p in probs.split(1, dim=1)]
        
        # Critic
        critic_states = critic_states.view(batch_size, num_frames, -1).mean(dim=1) # Average over frames
        value = self.critic_linear(critic_states)
        
        return dists, value

    def _forward_single(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)  # Flatten frames and batch
        
        # Shared feature extraction
        x_shared = self.shared_resnet(x)
        x_shared = F.adaptive_avg_pool2d(x_shared, (1, 1)).view(batch_size, num_frames, -1)  # Apply GAP and reshape

        return x_shared, x_shared

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, value, key, query):
        attention_output, _ = self.attention(query, key, value)
        output = self.fc_out(attention_output)
        return output
class ActorCritic(nn.Module):
    def __init__(self, num_frames, num_channels, hidden_size, height=0, width=0):
        super(ActorCritic, self).__init__()
        self.num_frames = num_frames
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
        self.critic_linear = nn.Linear(147456, 1) # [4,4,96,96]

    def forward(self, x):
        batch_size, _num_frames, channels, height, width = x.size()
        
        x = x.view(-1, channels, height, width)  # Reshape to [batch_size*num_frames, channels, height, width]
        
        # Actor
        x_actor = self.actor_conv(x)
        
        x_actor = x_actor.view(batch_size, _num_frames, -1)  # Reshape back to [batch_size, num_frames, features]
        _, (h_n, _) = self.actor_lstm(x_actor)
        action_logits = self.actor_linear(h_n.squeeze(0))
        action_logits = action_logits.view(batch_size, self.num_frames - 1, 5)  # Reshape to [batch_size, num_frames-1, 5]
        probs = F.softmax(action_logits, dim=-1)
        # print("prob:", probs.size())
        # print(probs.split(1, dim=1))
        dists = [Categorical(p) for p in probs.split(1, dim=1)]
        
        # Critic
        x_critic = self.critic_conv(x)
        x_critic = x_critic.view(batch_size, self.num_frames, -1).mean(dim=1)  # Average over frames
        value = self.critic_linear(x_critic)
        
        return dists, value

class ActorCritic_v2(nn.Module):
    def __init__(self, num_frames, hidden_size):
        super(ActorCritic_v2, self).__init__()
        
        # Shared ResNet Feature Extractor
        self.shared_resnet = torch.nn.Sequential(
            torch.nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Actor Network
        self.actor_lstm = torch.nn.LSTM(64, hidden_size, batch_first=True)
        self.actor_linear = torch.nn.Linear(hidden_size, 5 * (num_frames-1))
        self.num_frames = num_frames
        
        # Critic Network
        self.critic_linear = torch.nn.Linear(64, 1)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        actor_states, critic_states = self._forward_single(x)
        
        # Actor
        _, (h_n, _) = self.actor_lstm(actor_states)
        action_logits = self.actor_linear(h_n.squeeze(0))
        action_logits = action_logits.view(batch_size, self.num_frames - 1, 5)
        # print("what is x: ", x)
        # print("what is actor_states: ", actor_states)
        # print("what is h_n: ", h_n)
        probs = F.softmax(action_logits, dim=-1)
        # print("prob:", probs.size())
        # print(probs.split(1, dim=1))
        dists = [Categorical(p) for p in probs.split(1, dim=1)]
        # print("in DynamicActorCritic[tensor] [probs]: \n", probs.size())
        # Critic
        critic_states = critic_states.view(batch_size, num_frames, -1).mean(dim=1) # Average over frames
        value = self.critic_linear(critic_states)  
        
        return dists, value

    def _forward_single(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)  # Flatten frames and batch
        
        # Shared feature extraction
        x_shared = self.shared_resnet(x)
        # print("what is x_shared: ", x_shared)

        x_shared = F.adaptive_avg_pool2d(x_shared, (1, 1)).view(batch_size, num_frames, -1)  # Apply GAP and reshape

        return x_shared, x_shared

class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        # x: [batch_size, num_frames, feature_dim]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_weights = F.softmax(Q @ K.transpose(-2, -1) / (self.feature_dim ** 0.5), dim=-1)
        return attention_weights @ V

class ActorSAC(nn.Module):
    def __init__(self, num_frames, hidden_size):
        super(ActorSAC, self).__init__()

        # Initialize and modify ResNet for 4-channel input
        self.shared_resnet = models.resnet50(pretrained=False)
        self.modify_resnet_input_channels()

        self.attention = SelfAttention(2048)  # Assuming ResNet-50's feature size

        # Actor Network
        self.actor_lstm = nn.LSTM(2048, hidden_size, batch_first=True)
        self.actor_linear = nn.Linear(hidden_size, 5 * (num_frames-1))
        self.num_frames = num_frames

    def modify_resnet_input_channels(self):
        original_conv = self.shared_resnet.conv1
        new_conv = nn.Conv2d(4, original_conv.out_channels, 
                            kernel_size=original_conv.kernel_size, 
                            stride=original_conv.stride, 
                            padding=original_conv.padding,
                            bias=original_conv.bias is not None)
        # Copy weights from original layer to new layer
        with torch.no_grad():
            new_conv.weight[:, :3] = original_conv.weight
            new_conv.weight[:, 3] = original_conv.weight[:, 0]
        self.shared_resnet.conv1 = new_conv
        
        # Remove the fully connected layer
        self.shared_resnet = nn.Sequential(*list(self.shared_resnet.children())[:-2])


    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        actor_states = self._forward_single(x)

        # Actor
        _, (h_n, _) = self.actor_lstm(actor_states)
        action_logits = self.actor_linear(h_n.squeeze(0))
        action_logits = action_logits.view(batch_size, self.num_frames - 1, 5)

        probs = F.softmax(action_logits, dim=-1)
        dists = [Categorical(p) for p in probs.split(1, dim=1)]

        return dists

    def _forward_single(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        
        # Process each frame separately and store features in a list
        features = []
        for i in range(num_frames):
            frame = x[:, i, :, :, :]
            # print(frame.shape)
            frame_features = self.shared_resnet(frame)
            
            # time.sleep(1000)
            frame_features = F.adaptive_avg_pool2d(frame_features, (1, 1)).squeeze(-1).squeeze(-1)  # shape: [batch_size, 2048]
            features.append(frame_features)

        # Stack features along the new dimension
        features_stacked = torch.stack(features, dim=1)  # shape: [batch_size, num_frames, 2048]

        # Attention over the frames
        aggregated_features = self.attention(features_stacked)

        return aggregated_features

class qValueNetwork(nn.Module):
    def __init__(self, num_frames):
        super(qValueNetwork, self).__init__()

        # Shared ResNet Feature Extractor
        self.shared_resnet = models.resnet50(pretrained=False)
        self.modify_resnet_input_channels()

        # Temporal Attention Mechanism
        self.temporal_attention = SelfAttention(2048)

        # Temporal LSTM layer
        self.lstm = nn.LSTM(2048, 1024, batch_first=True)
        
        # Value prediction for each frame
        self.value_head = nn.Linear(1024, 5 * (num_frames - 1))
        
        self.transform_layer = nn.Linear(60, 15)

    def modify_resnet_input_channels(self):
        original_conv = self.shared_resnet.conv1
        new_conv = nn.Conv2d(4, original_conv.out_channels, 
                            kernel_size=original_conv.kernel_size, 
                            stride=original_conv.stride, 
                            padding=original_conv.padding,
                            bias=original_conv.bias is not None)
        # Copy weights from original layer to new layer
        with torch.no_grad():
            new_conv.weight[:, :3] = original_conv.weight
            new_conv.weight[:, 3] = original_conv.weight[:, 0]
        self.shared_resnet.conv1 = new_conv
        
        # Remove the fully connected layer
        self.shared_resnet = nn.Sequential(*list(self.shared_resnet.children())[:-2])

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)  # Flatten frames and batch
        # print(f"Debug x shape: {x.size()}")
        # Shared feature extraction
        spatial_features = self.shared_resnet(x)
        # print(f"Debug spatial_features shape: {spatial_features.size()}")
        # Global Average Pooling (GAP)
        spatial_features = F.adaptive_avg_pool2d(spatial_features, (1, 1))
        # print(f"Debug spatial_features shape: {spatial_features.size()}")
        spatial_features = spatial_features.view(batch_size, num_frames, -1)
        # print(f"Debug spatial_features shape: {spatial_features.size()}")
        # Temporal Attention
        temporal_features = self.temporal_attention(spatial_features)
        # print(f"Debug temporal_features shape: {temporal_features.size()}")
        # Temporal LSTM layer
        lstm_out, _ = self.lstm(temporal_features)
        # print(f"Debug lstm_out shape: {lstm_out.size()}")
        # Value prediction
        values = self.value_head(lstm_out)
        # print(f"Debug values shape: {values.size()}")
        values = values.view(batch_size, -1)
        
        transformed_values = self.transform_layer(values)
        transformed_values = transformed_values.view(batch_size, num_frames-1, 5)
        

        return transformed_values


from torchvision.models import resnet18

class DynamicActorCritic(nn.Module):
    def __init__(self, hidden_size):
        super(DynamicActorCritic, self).__init__()
        
        # Shared ResNet Feature Extractor
        self.shared_resnet = torch.nn.Sequential(
            torch.nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Actor Network
        self.actor_lstm = torch.nn.LSTM(64, hidden_size, batch_first=True)
        self.actor_linear = torch.nn.Linear(hidden_size, 15)
        
        # Critic Network
        self.critic_linear = torch.nn.Linear(64, 1)

    def forward(self, x):
        if isinstance(x, list):  # If x is a list, handle each element separately
            actor_dists = []
            critic_values = []
            for idx, single_x in enumerate(x):
                single_x = single_x.unsqueeze(0)  # Add a batch dimension
                # print("in DynamicActorCritic[list] [single_x]: \n", single_x.size())
                actor_state, critic_state = self._forward_single(single_x)
                # print("in DynamicActorCritic[list] [actor_state]: \n", actor_state.size())
                # print("in DynamicActorCritic[list] [critic_state]: \n", critic_state.size())
                # Actor
                _, (h_n, _) = self.actor_lstm(actor_state)
                action_logits = self.actor_linear(h_n.squeeze(0))
                probs = F.softmax(action_logits, dim=-1)
                # print("in DynamicActorCritic[list] [probs]: \n", probs.size())
                dists = Categorical(probs)
                
                actor_dists.append(dists)
                
                # Critic
                value = self.critic_linear(critic_state.mean(dim=1))
                value = value.squeeze(0) # Average over frames
                critic_values.append(value)
            
            return actor_dists, critic_values
        else:  # x is a tensor
            actor_states, critic_states = self._forward_single(x)
            
            # Actor
            _, (h_n, _) = self.actor_lstm(actor_states)
            action_logits = self.actor_linear(h_n.squeeze(0))
            probs = F.softmax(action_logits, dim=-1)
            dists = Categorical(probs)
            # print("in DynamicActorCritic[tensor] [probs]: \n", probs.size())
            # Critic
            value = self.critic_linear(critic_states.mean(dim=1))  # Average over frames
            
            return dists, value

    def _forward_single(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)  # Flatten frames and batch
        
        # Shared feature extraction
        x_shared = self.shared_resnet(x)
        x_shared = F.adaptive_avg_pool2d(x_shared, (1, 1)).view(batch_size, num_frames, -1)  # Apply GAP and reshape

        return x_shared, x_shared

class DynamicActorCriticWithSize(nn.Module):
    def __init__(self, hidden_size):
        super(DynamicActorCriticWithSize, self).__init__()
        
        # Shared ResNet Feature Extractor
        self.shared_resnet = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Actor Network
        self.actor_lstm = nn.LSTM(65, hidden_size, batch_first=True)  # 65 due to concatenation with sizes
        self.actor_linear = nn.Linear(hidden_size, 15)
        
        # Critic Network
        self.critic_linear = nn.Linear(65, 1)  # 65 due to concatenation with sizes

    def forward(self, x, sizes):
        if isinstance(x, list):  # If x is a list, handle each element separately
            actor_dists = []
            critic_values = []
            for idx, single_x in enumerate(x):
                single_size = sizes[idx].unsqueeze(0).repeat(1, single_x.size(0), 1)  # Expand size for concatenation
                actor_state, critic_state = self._forward_single(single_x.unsqueeze(0), single_size)
                
                # Actor
                _, (h_n, _) = self.actor_lstm(actor_state)
                action_logits = self.actor_linear(h_n.squeeze(0))
                probs = F.softmax(action_logits, dim=-1)
                dists = Categorical(probs)
                actor_dists.append(dists)
                
                # Critic
                value = self.critic_linear(critic_state.mean(dim=1))  # Average over frames
                value = value.squeeze(0)
                critic_values.append(value)
            
            return actor_dists, critic_values
        else:  # x is a tensor
            # print("sizes size: ", sizes.size())
            sizes = sizes.unsqueeze(2).repeat(1, x.size(1), 1)  # Expand size for concatenation
            actor_states, critic_states = self._forward_single(x, sizes)
            
            # Actor
            _, (h_n, _) = self.actor_lstm(actor_states)
            action_logits = self.actor_linear(h_n.squeeze(0))
            probs = F.softmax(action_logits, dim=-1)
            dists = Categorical(probs)
            
            # Critic
            value = self.critic_linear(critic_states.mean(dim=1))  # Average over frames
            
            return dists, value

    def _forward_single(self, x, sizes):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)  # Flatten frames and batch
        
        # Shared feature extraction
        x_shared = self.shared_resnet(x)
        x_shared = F.adaptive_avg_pool2d(x_shared, (1, 1)).view(batch_size, num_frames, -1)  # Apply GAP and reshape
        
        # Concatenate size

        x_shared = torch.cat([x_shared, sizes], dim=2)
        
        return x_shared, x_shared
class DynamicActorCriticWithSizeAttention(nn.Module):
    def __init__(self, hidden_size):
        super(DynamicActorCriticWithSizeAttention, self).__init__()
        
        # Shared ResNet-like Feature Extractor
        self.shared_resnet = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.attention = MultiHeadAttention(64, 64, num_heads=4)
        
        # Actor Network
        self.actor_lstm = nn.LSTM(64, hidden_size, batch_first=True)
        self.actor_linear = nn.Linear(hidden_size, 15)
        
        # Critic Network
        self.critic_linear = nn.Linear(64, 1)

    def forward(self, x, sizes):
        if isinstance(x, list):  # If x is a list, handle each element separately
            actor_dists = []
            critic_values = []
            for idx, single_x in enumerate(x):
                single_size = sizes[idx].unsqueeze(0).unsqueeze(1).repeat(1, single_x.size(1), 1)
                single_x = single_x.unsqueeze(0)  # Add a batch dimension
                actor_state, critic_state = self._forward_single(single_x, single_size)
                
                # Actor
                _, (h_n, _) = self.actor_lstm(actor_state)
                action_logits = self.actor_linear(h_n.squeeze(0))
                probs = F.softmax(action_logits, dim=-1)
                dists = Categorical(probs)
                actor_dists.append(dists)
                
                # Critic
                value = self.critic_linear(critic_state.mean(dim=1))  # Average over frames
                critic_values.append(value)
            
            return actor_dists, critic_values
        else:  # x is a tensor
            actor_states, critic_states = self._forward_single(x, sizes)
            
            # Actor
            _, (h_n, _) = self.actor_lstm(actor_states)
            action_logits = self.actor_linear(h_n.squeeze(0))
            probs = F.softmax(action_logits, dim=-1)
            dists = Categorical(probs)
            
            # Critic
            value = self.critic_linear(critic_states.mean(dim=1))  # Average over frames
            
            return dists, value

    def _forward_single(self, x, sizes):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)  # Flatten frames and batch
        
        # Shared feature extraction
        x_shared = self.shared_resnet(x)
        x_shared = F.adaptive_avg_pool2d(x_shared, (1, 1)).view(batch_size, num_frames, -1)  # Apply GAP and reshape
        
        # Attention mechanism to combine size information
        sizes = sizes.unsqueeze(0)  # Add sequence length dimension
        x_shared = self.attention(x_shared, x_shared, sizes)
        
        return x_shared, x_shared
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
        # print("!!!!!!!!!!!!!!!!policynet and sr_net's device: ", device)
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
        # print("!!!!!!!!!!!!!!!!net and sca's device: ", device)
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

# Define a simple test case function
def test_DynamicActorCritic(model):
    # Test case 1: Tensor input
    x_tensor = torch.randn(2, 3, 4, 96, 96)
    sizes_tensor = torch.randn(2, 1)
    dists1, value1 = model(x_tensor, sizes_tensor)
    print("Test case 1 (tensor input) - Distributions shape:", dists1.probs.size())
    print("Test case 1 (tensor input) - Value shape:", value1.size())

    # Test case 2: List input with varying burst sizes
    x_list = [torch.randn(3, 4, 96, 96), torch.randn(4, 4, 96, 96)]
    sizes_list = torch.randn(2, 1)
    dists2, value2 = model(x_list, sizes_list)
    print("Test case 2 (list input) - Distributions shape:", [dist.probs.size() for dist in dists2])
    print("Test case 2 (list input) - Value shape:", [value.size() for value in value2])



if __name__ == '__main__':
    # Initialize the model
    hidden_size = 128
    model = DynamicActorCriticWithSize(hidden_size)
    # Run the test cases
    test_DynamicActorCritic(model)
    