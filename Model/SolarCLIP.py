import torch
import torch.nn as nn
from typing import Tuple, Union
import numpy as np
from einops import einsum

#for train
from .image_encoder import VisionTransformer,LayerNorm

#for test in this dir
#from image_encoder import VisionTransformer,LayerNorm
#from txt_encoder import TextTransformer


class SolarCLIP_MODEL(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # mag vision
                 image_resolution_mag: int,
                 vision_layers_mag: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size_mag: int,
                 # 11 channels
                 image_resolution_H: int,
                 vision_layers_H: Union[Tuple[int, int, int, int], int],
                 #vision_width: int,
                 vision_patch_size_H: int,
                 # loss token type
                 transformer_token_type: str,
                 ):
        super().__init__()


        vision_heads = vision_width // 64
        self.visual_mag = VisionTransformer(
                in_channels=1,
                input_resolution=image_resolution_mag,
                patch_size=vision_patch_size_mag,
                width=vision_width,
                layers=vision_layers_mag,
                heads=vision_heads,
                output_dim=embed_dim,
                token_type = transformer_token_type
            )
        
        self.visual_H = VisionTransformer(
                in_channels=1,
                input_resolution=image_resolution_H,
                patch_size=vision_patch_size_H,
                width=vision_width,
                layers=vision_layers_H,
                heads=vision_heads,
                output_dim=embed_dim,
                token_type = transformer_token_type

            )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # parameters for weighted loss
        self.transformer_token_type = transformer_token_type


        #self.initialize_parameters()

   
    
        
    @property
    def dtype(self):
        return self.visual_mag.conv1.weight.dtype

    def encode_mag(self, image_mag):
        return self.visual_mag(image_mag.type(self.dtype))

    def encode_H(self, image_H):
        return self.visual_H(image_H.type(self.dtype))

    def forward(self, image_mag, image_H, token_weight_1=None, token_weight_2=None):

        
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

            if token_weight_1 is None:
                B = mag_features.shape[0]
                L = mag_features.shape[1]
                token_weight_1 = torch.ones([B,L],dtype = mag_features.dtype,device = mag_features.device)
            if token_weight_2 is None:
                B = H_features.shape[0]
                L = H_features.shape[1]
                token_weight_2 = torch.ones([B,L],dtype = H_features.dtype,device = H_features.device)
            assert (token_weight_1.shape == (B, L) and token_weight_2.shape == (B, L)) # [B,L] tensor
            token_weight_1 = token_weight_1.unsqueeze(-1)
            token_weight_2 = token_weight_2.unsqueeze(-1)

            mag_features = torch.einsum('BLD,BLd->BLD', mag_features, token_weight_1)
            H_features = torch.einsum('BLD,BLd->BLD', H_features, token_weight_2)
            inner_cor_matrix = torch.einsum('BLD,BlD->BLl', mag_features, H_features)
            cor_matrix = torch.einsum('BLD,bLD->BbL', mag_features, H_features)
            cor_matrix = cor_matrix.mean(dim=-1) # [B,B]

            logits_per_mag = logit_scale * cor_matrix
            logits_per_H = logits_per_mag.t()

        # shape = [global_batch_size, global_batch_size]
        # return logits_per_mag, logits_per_H
        return logits_per_mag, logits_per_H, inner_cor_matrix
    

def get_model_from_args(args):
    return SolarCLIP_MODEL(
            embed_dim = args.embed_dim,
            image_resolution_mag = args.image_resolution_mag,
            vision_layers_mag = args.vision_layers_mag,
            vision_width = args.vision_width,
            vision_patch_size_mag = args.vision_patch_size_mag,
            image_resolution_H = args.image_resolution_H,
            vision_layers_H = args.vision_layers_H,
            vision_patch_size_H = args.vision_patch_size_H,
            transformer_token_type = args.token_type
        )