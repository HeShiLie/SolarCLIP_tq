from typing import Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from Modules.clip_modules.vit import VisionTransformer, Transformer, Remove_class_token
from Model.models import PatchMerging, PatchExpand, Decoder

"""
SolarCLIP_pos_VAE Model.
This CLIP model trained on the latent space of the VAE model.
"""

class SolarCLIP_pos_VAE(nn.Module):
    def __init__(self,
                 embed_dim: int=768,
                 # mag vision
                 image_resolution_mag: int=64,
                 vision_layers_mag: Union[Tuple[int, int, int, int], int]=12,
                 vision_width: int=768,
                 vision_patch_size_mag: int=4,
                 # 11 channels
                 image_resolution_H: int=64,
                 vision_layers_H: Union[Tuple[int, int, int, int], int]=12,
                 #vision_width: int,
                 vision_patch_size_H: int=4,
                 # loss token type
                 transformer_token_type: str='all embedding',
                 norm_type: str='bn1d'
                 ):
        super().__init__()

        vision_heads = vision_width // 64
        self.visual_mag = VisionTransformer(
                in_channels=3,
                input_resolution=image_resolution_mag,
                patch_size=vision_patch_size_mag,
                width=vision_width,
                layers=vision_layers_mag,
                heads=vision_heads,
                output_dim=embed_dim,
                token_type = transformer_token_type,
                norm_type= norm_type
            )
        
        self.visual_H = VisionTransformer(
                in_channels=3,
                input_resolution=image_resolution_H,
                patch_size=vision_patch_size_H,
                width=vision_width,
                layers=vision_layers_H,
                heads=vision_heads,
                output_dim=embed_dim,
                token_type = transformer_token_type,
                norm_type = norm_type
            )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # parameters for weighted loss
        self.transformer_token_type = transformer_token_type

        #self.initialize_parameters()

    @property
    def dtype(self):
        # return self.visual_mag.conv1.weight.dtype
        return self.visual_mag.conv1.conv1.weight.dtype

    def encode_mag(self, image_mag):
        return self.visual_mag(image_mag.type(self.dtype))

    def encode_H(self, image_H):
        return self.visual_H(image_H.type(self.dtype))

    def forward(self, image_mag, image_H, token_weight_1=None, token_weight_2=None):
        """
        image_mag: [batch_size, 3, 64, 64]
        image_H: [batch_size, 3, 64, 64]
        token_weight_1: [batch_size, num_patches] or None
        token_weight_2: [batch_size, num_patches] or None
        """
        mag_features = self.encode_mag(image_mag)   #shape = [batch_size, length,embed_dim]
        H_features = self.encode_H(image_H)
        
        # normalized features
        mag_features = mag_features / (mag_features.norm(dim=-1, keepdim=True)+1e-32)
        H_features = H_features / (H_features.norm(dim=-1, keepdim=True)+1e-32)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()

        if self.transformer_token_type == 'class embedding':
            logits_per_mag = logit_scale * mag_features @ H_features.t()
            logits_per_H = logits_per_mag.t()
            inner_cor_matrix = None

        elif self.transformer_token_type == 'all embedding':
            B = mag_features.shape[0]
            L = mag_features.shape[1]
            if token_weight_1 is None:
                token_weight_1 = torch.ones([B,L],dtype = mag_features.dtype,device = mag_features.device)
            if token_weight_2 is None:
                token_weight_2 = torch.ones([B,L],dtype = H_features.dtype,device = H_features.device)
            assert (token_weight_1.shape == (B, L) and token_weight_2.shape == (B, L)) # [B,L] tensor
            token_weight_1 = token_weight_1.unsqueeze(-1)
            token_weight_2 = token_weight_2.unsqueeze(-1)

            mag_features = torch.einsum('BLD,BLd->BLD', mag_features, token_weight_1)
            H_features = torch.einsum('BLD,BLd->BLD', H_features, token_weight_2)
            inner_cor_matrix = torch.einsum('BLD,BlD->BLl', mag_features, H_features)
            inner_cor_matrix = inner_cor_matrix.mean(dim=0) # [L,L]
            cor_matrix = torch.einsum('BLD,bLD->BbL', mag_features, H_features)
            cor_matrix = cor_matrix.mean(dim=-1) # [B,B]

            logits_per_mag = logit_scale * cor_matrix
            logits_per_H = logits_per_mag.t()
            inner_cor_matrix = logit_scale * inner_cor_matrix

        # shape = [global_batch_size, global_batch_size]
        # return logits_per_mag, logits_per_H
        return logits_per_mag, logits_per_H, inner_cor_matrix
    
    def calculate_loss(self, mag_image, h_image, inner_loss_rate = 0, token_weight_1 = None, token_weight_2 = None, criterion = torch.nn.functional.cross_entropy):

        logits_per_mag, logits_per_h, inner_cor_matrix = self.forward(mag_image, h_image, token_weight_1, token_weight_2)
        ground_truth = torch.arange(len(mag_image), dtype=torch.long, device=mag_image.device)
        
        loss_img = criterion(logits_per_mag, ground_truth)
        loss_h = criterion(logits_per_h, ground_truth)
        loss = (loss_img + loss_h) / 2
        acc = (torch.argmax(logits_per_mag, dim=1) == ground_truth).float().mean().item()

        assert inner_loss_rate >=0
        if inner_loss_rate > 0:
            ground_truth = torch.arange(inner_cor_matrix.shape[-1], dtype=torch.long, device=inner_cor_matrix.device)#[l]
            # ground_truth = ground_truth.unsqueeze(0).expand(inner_cor_matrix.shape[0],-1) # [L,L]
            loss_inner = criterion(inner_cor_matrix,ground_truth)/2
            loss_inner = loss_inner + criterion(inner_cor_matrix.t(),ground_truth)/2
        else:
            loss_inner = torch.tensor(0, dtype=torch.float32, device=inner_cor_matrix.device)

        return loss, loss_inner, acc, logits_per_mag, inner_cor_matrix
    
def get_SolarCLIP_pos_VAEfrom_args(args):
    return SolarCLIP_pos_VAE(
        in_channels = args.in_channels, 
        input_resolution = args.input_resolution, 
        patch_size = args.patch_size, 
        width = args.width, 
        vit_layers = args.vit_layers,
        transformer_layers = args.transformer_layers, 
        hidden_dim = args.hidden_dim, 
        token_type = args.token_type, 
        norm_type = args.norm_type,
        output_dim = args.output_dim,
        output_size = args.output_size,
        deprojection_type = args.deprojection_type,
        with_bias = args.with_bias,
        loss_type = args.loss_type,
        lambda_kl = args.lambda_kl)