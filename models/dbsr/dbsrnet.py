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
import torch
import torch.nn as nn
import torch.nn.functional as F
import models_dbsr.dbsr.encoders as dbsr_encoders
import models_dbsr.dbsr.decoders as dbsr_decoders
import models_dbsr.dbsr.merging as dbsr_merging
from admin.model_constructor import model_constructor
from models_dbsr.alignment.pwcnet import PWCNet
from admin.environment import env_settings

class PolicyNet(nn.Module):
    def __init__(self, out_dim=4, num_actions=5):
        super(PolicyNet, self).__init__()
        self.conv = nn.Conv2d(out_dim, 64, kernel_size=3, padding=1)
        self.fc_action1 = nn.Conv2d(64, num_actions, kernel_size=1)
        self.fc_action2 = nn.Conv2d(64, num_actions, kernel_size=1)
        self.fc_action3 = nn.Conv2d(64, num_actions, kernel_size=1)
        self.num_actions = num_actions

    def forward(self, x):    
        # print("x size1: ", x.size())
        x = F.relu(self.conv(x))
        # print("x size2: ", x.size())
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # x = x.view(x.size(0), -1)
        # print("x size3: ", x.size())
        x1 = self.fc_action1(x)
        x2 = self.fc_action2(x)
        x3 = self.fc_action3(x)
        # print("x size4: ", x.size())
        action1_logits = x1.squeeze(-1).squeeze(-1).view(-1, self.num_actions)
        action2_logits = x2.squeeze(-1).squeeze(-1).view(-1, self.num_actions)
        action3_logits = x3.squeeze(-1).squeeze(-1).view(-1, self.num_actions)
        actions_logits = torch.stack([action1_logits, action2_logits, action3_logits], dim=1) #[batch_size, three actions for three shifted images, num_actions]
        
        # print("x size5: ", actions_logits.size())
        # time.sleep(1000)

        return actions_logits

class PolicyNet_v2(nn.Module):
    def __init__(self, out_dim=4, num_actions=3):
        super(PolicyNet_v2, self).__init__()
        self.conv = nn.Conv2d(out_dim, 64, kernel_size=3, padding=1)
        self.fc_action1 = nn.Conv2d(64, num_actions, kernel_size=1)
        self.fc_action2 = nn.Conv2d(64, num_actions, kernel_size=1)
        self.fc_action3 = nn.Conv2d(64, num_actions, kernel_size=1)
        self.num_actions = num_actions

    def forward(self, x):    
        # print("x size1: ", x.size())
        x = F.relu(self.conv(x))
        # print("x size2: ", x.size())
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # x = x.view(x.size(0), -1)
        # print("x size3: ", x.size())
        x1 = self.fc_action1(x)
        x2 = self.fc_action2(x)
        x3 = self.fc_action3(x)
        # print("x size4: ", x.size())
        action1_logits = x1.squeeze(-1).squeeze(-1).view(-1, self.num_actions)
        action2_logits = x2.squeeze(-1).squeeze(-1).view(-1, self.num_actions)
        action3_logits = x3.squeeze(-1).squeeze(-1).view(-1, self.num_actions)
        actions_logits = torch.stack([action1_logits, action2_logits, action3_logits], dim=1) #[batch_size, three actions for three shifted images, num_actions]
        
        # print("x size5: ", actions_logits.size())
        # time.sleep(1000)

        return actions_logits

class DBSRNet(nn.Module):
    """ Deep Burst Super-Resolution model"""
    def __init__(self, encoder, merging, decoder):
        super().__init__()

        self.encoder = encoder      # Encodes input images and performs alignment
        self.merging = merging      # Merges the input embeddings to obtain a single feature map
        self.decoder = decoder      # Decodes the merged embeddings to generate HR RGB image

    def forward(self, im):
        out_enc = self.encoder(im)
        out_merge = self.merging(out_enc)
        out_dec = self.decoder(out_merge)

        return out_dec['pred'], {'offsets': out_enc['offsets'], 'fusion_weights': out_merge['fusion_weights']}


@model_constructor
def dbsrnet_cvpr2021(enc_init_dim, enc_num_res_blocks, enc_out_dim,
                     dec_init_conv_dim, dec_num_pre_res_blocks, dec_post_conv_dim, dec_num_post_res_blocks,
                     upsample_factor=2, activation='relu', train_alignmentnet=False,
                     offset_feat_dim=64,
                     weight_pred_proj_dim=32,
                     num_offset_feat_extractor_res=1,
                     num_weight_predictor_res=1,
                     offset_modulo=1.0,
                     use_offset=True,
                     ref_offset_noise=0.0,
                     softmax=True,
                     use_base_frame=True,
                     icnrinit=False,
                     gauss_blur_sd=None,
                     gauss_ksz=3,
                     use_pretrained=None,
                     with_attention=False
                     ):
    # backbone
    alignment_net = PWCNet(load_pretrained=True,
                           weights_path='{}/pwcnet-network-default.pth'.format(env_settings().pretrained_nets_dir))

    # print("pwcnet's device!!!!!!!!!!!!!: ", next(alignment_net.parameters()).device)

    encoder = dbsr_encoders.ResEncoderWarpAlignnet(enc_init_dim, enc_num_res_blocks, enc_out_dim,
                                                   alignment_net,
                                                   activation=activation,
                                                   train_alignmentnet=train_alignmentnet,
                                                   with_attention=with_attention)
    # print("encoder's device!!!!!!!!!!!!!: ", next(encoder.parameters()).device)
 
    

    merging = dbsr_merging.WeightedSum(enc_out_dim, weight_pred_proj_dim, offset_feat_dim,
                                       num_offset_feat_extractor_res=num_offset_feat_extractor_res,
                                       num_weight_predictor_res=num_weight_predictor_res,
                                       offset_modulo=offset_modulo,
                                       use_offset=use_offset,
                                       ref_offset_noise=ref_offset_noise,
                                       softmax=softmax, use_base_frame=use_base_frame)

    decoder = dbsr_decoders.ResPixShuffleConv(enc_out_dim, dec_init_conv_dim, dec_num_pre_res_blocks,
                                              dec_post_conv_dim, dec_num_post_res_blocks,
                                              upsample_factor=upsample_factor, activation=activation,
                                              gauss_blur_sd=gauss_blur_sd, icnrinit=icnrinit,
                                              gauss_ksz=gauss_ksz)

    net = DBSRNet(encoder=encoder, merging=merging, decoder=decoder)
    if use_pretrained is not None:
        pretrained_weights = torch.load(use_pretrained)
        net.load_state_dict(pretrained_weights)
    return net
