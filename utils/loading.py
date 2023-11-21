import time
import os
import admin.loading as loading
import torch
from admin.environment import env_settings


# def load_network(net_path, modify_first_layer=False, return_dict=False, net_init=None, **kwargs):
#     kwargs['backbone_pretrained'] = False
#     if os.path.isabs(net_path):
#         path_full = net_path
#         net, checkpoint_dict = loading.load_network(path_full, **kwargs)
#     else:
#         path_full = os.path.join(env_settings().workspace_dir, 'checkpoints', net_path)
#         net, checkpoint_dict = loading.load_network(path_full, **kwargs)
    
#     if modify_first_layer:
#         # Load the entire checkpoint
#         checkpoint = torch.load(path_full, map_location='cpu')
#         checkpoint_dict = checkpoint['net']

#         # The key for the first layer's weights in the state_dict
#         first_layer_key = 'lr_encoder.init_layer.0.weight'

#         # Load only lr_encoder and hr_decoder weights from checkpoint
#         model_dict = net_init.state_dict()
#         pretrained_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict and k != first_layer_key}
#         hr_decoder_dict = {k: v for k, v in checkpoint_dict.items() if k.startswith('hr_decoder')}



#         # Load the pretrained weights for all layers except the first layer
#         model_dict = net_init.state_dict()
#         # pretrained_dict = {k: v for k, v in checkpoint_dict['net'].items() if k in model_dict and k != first_layer_key}
#         model_dict.update(pretrained_dict)
#         net_init.load_state_dict(model_dict)

#         # Handle the first layer weights: average the weights across the input channels
#         if first_layer_key in checkpoint_dict:
#             first_layer_weights = checkpoint_dict[first_layer_key]
#             # Average across the input channels dimension (assuming it's the first dimension after batch)
#             new_first_layer_weights = first_layer_weights.mean(dim=1, keepdim=True)
#             # Assign the modified weights to the first layer of the network
#             getattr(net, 'lr_encoder').init_layer[0].weight.data = new_first_layer_weights
#         # Load the pretrained weights for all layers
#         model_dict = net_init.state_dict()
#         pretrained_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict}
#         model_dict.update(pretrained_dict)
#         net_init.load_state_dict(model_dict)

#         # Modify the last layer of hr_decoder for 1-channel output
#         last_layer_key = 'hr_decoder.predictor.0.weight'  # This key needs to be verified
#         last_layer_bias_key = 'hr_decoder.predictor.0.bias'  # This key needs to be verified
        
#         # Handle the last layer weights: average the weights across the output channels
#         if last_layer_key in checkpoint_dict:
#             last_layer_weights = checkpoint_dict[last_layer_key]
#             # Average across the output channels dimension (assuming it's the second dimension for Conv2d)
#             new_last_layer_weights = last_layer_weights.mean(dim=0, keepdim=True).expand(-1, *last_layer_weights.shape[1:])
#             # Assign the modified weights to the last layer of the network
#             getattr(net, 'hr_decoder').predictor[0].weight.data = new_last_layer_weights
#             getattr(net, 'hr_decoder').predictor[0].bias.data = checkpoint_dict[last_layer_bias_key]

#         # Update the model's state dict with pretrained weights
#         model_dict.update(hr_decoder_dict)
#         net_init.load_state_dict(model_dict, strict=False)
    
#     net = net_init  

#     if return_dict:
#         return net, checkpoint_dict
#     else:
#         return net

def load_network(net_path, modify_first_layer=False, use_all_pretrained_weights=True, net_init=None, **kwargs):
    checkpoint = torch.load(net_path, map_location='cpu')
    checkpoint_dict = checkpoint['net']

    # Filter out weights for lr_encoder and hr_decoder
    
    if net_init is None:
        if os.path.isabs(net_path):
            path_full = net_path
            net, checkpoint_dict = loading.load_network(path_full, **kwargs)
        else:
            path_full = os.path.join(env_settings().workspace_dir, 'checkpoints', net_path)
            net, checkpoint_dict = loading.load_network(path_full, **kwargs)
        if 'return_dict' in kwargs and kwargs['return_dict']:
            return net, checkpoint_dict
        else:
            return net
    if modify_first_layer:
        pretrained_dict = {k: v for k, v in checkpoint_dict.items() if 'lr_encoder' in k or 'hr_decoder' in k}
        # Modify the first layer of lr_encoder for 1-channel input
        first_layer_key = 'lr_encoder.init_layer.0.weight'
        if first_layer_key in pretrained_dict:
            first_layer_weights = pretrained_dict[first_layer_key]
            new_first_layer_weights = first_layer_weights.mean(dim=1, keepdim=True)
            pretrained_dict[first_layer_key] = new_first_layer_weights.repeat(1, 1, 1, 1)

        # Modify the last layer of hr_decoder for 1-channel output
        last_layer_key = 'hr_decoder.predictor.0.weight'
        if last_layer_key in pretrained_dict:
            last_layer_weights = pretrained_dict[last_layer_key]
            new_last_layer_weights = last_layer_weights.mean(dim=0, keepdim=True).expand(-1, *last_layer_weights.shape[1:])
            pretrained_dict[last_layer_key] = new_last_layer_weights
            # Adjust bias if it exists
            last_layer_bias_key = 'hr_decoder.predictor.0.bias'
            if last_layer_bias_key in pretrained_dict:
                new_last_layer_bias = pretrained_dict[last_layer_bias_key].mean().unsqueeze(0)
                pretrained_dict[last_layer_bias_key] = new_last_layer_bias
    
    elif use_all_pretrained_weights:
        pretrained_dict = {k: v for k, v in checkpoint_dict.items()}
    else:
        pretrained_dict = {k: v for k, v in checkpoint_dict.items() if 'lr_encoder' in k or 'hr_decoder' in k}

    # Update the model with the new state dict
    model_dict = net_init.state_dict()
    model_dict.update(pretrained_dict)
    net_init.load_state_dict(model_dict, strict=False)


    if 'return_dict' in kwargs and kwargs['return_dict']:
        return net_init, checkpoint
    else:
        return net_init
