import torch
import torch.nn as nn
from typing import Tuple, Union
import numpy as np
from einops import einsum, rearrange

#for train
from .vit import VisionTransformer, Decoder, PatchMerging, PatchExpand, Transformer, Remove_class_token
# from vit import VisionTransformer, Decoder, PatchMerging, PatchExpand, Transformer, Remove_class_token


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
                 norm_type: str 
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
                token_type = transformer_token_type,
                norm_type= norm_type
            )
        
        self.visual_H = VisionTransformer(
                in_channels=1,
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
        # return loss, acc, logits_per_mag
    
    
class SolarReconModel(nn.Module):
    def __init__(self, 
                 in_channels : int = 1, 
                 input_resolution: int = 1024, 
                 patch_size: int = 64, 
                 width: int = 768, 
                 layers: int = 12, 
                 hidden_dim: int = 768, 
                 token_type: str = 'all_embedding', 
                 norm_type: str = 'bn1d',
                 output_dim: int = 1,
                 output_size: int = 1024,
                 deprojection_type: str = 'linear',
                 with_bias: bool = True):
        super().__init__()
        heads = width // 64
        self.vit_model = VisionTransformer(
            in_channels = in_channels,
            input_resolution = input_resolution, 
            patch_size = patch_size, 
            width = width, 
            layers = layers, 
            heads = heads, 
            output_dim = hidden_dim,
            token_type = token_type, 
            norm_type = norm_type)
        self.recon = Decoder(
            embed_dim = hidden_dim, 
            output_dim = output_dim, 
            output_size = output_size, 
            deprojection_type = deprojection_type, 
            with_bias = with_bias
        )

    def forward(self, x):
        features = self.vit_model(x) # [B,L,D]
        features = features[:,1:,:] # remove class token
        recon = self.recon(features)
        return recon
    
    def calculate_loss(self, image, weights, criterion = nn.functional.mse_loss):
        recon = self(image)
        mse = criterion(recon, image)
        mse_with_weight = weights*criterion(recon, image, reduction='none') 
        mse_with_weight = mse_with_weight.mean()
        return mse_with_weight, mse

class SolarReconModel_Unet_like(nn.Module):
    def __init__(self,
                 in_channels : int = 1, 
                 input_resolution: int = 1024, 
                 patch_size: int = 64, 
                 width: int = 768, 
                 vit_layers: int = 12,
                 transformer_layers: int = 2, 
                 hidden_dim: int = 768, 
                 token_type: str = 'all_embedding', 
                 norm_type: str = 'bn1d',
                 output_dim: int = 1,
                 output_size: int = 1024,
                 deprojection_type: str = 'linear',
                 with_bias: bool = True):
        super().__init__()
        heads = width // patch_size
        self.vit_model = VisionTransformer(
            in_channels = in_channels,
            input_resolution = input_resolution, 
            patch_size = patch_size, 
            width = width, 
            layers = vit_layers, 
            heads = heads, 
            output_dim = hidden_dim,
            token_type = token_type, 
            norm_type = norm_type)
        self.transfomer = Transformer
        self.PatchMerging = PatchMerging
        self.PatchExpanding = PatchExpand
        self.Remove_class_token = Remove_class_token
        
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.patch_length = (self.input_resolution // self.patch_size) ** 2
        self.hidden_dim = hidden_dim
        self.layers = transformer_layers

        # self.encoder = nn.Sequential(
        #     self.vit_model, # (B, C, H, W) -> (B, L, D) (B,16*16+1,768)
        #     self.Remove_class_token(), #(B, L+1, D) -> (B, L, D)
        #     self.PatchMerging(input_length = self.patch_length, dim = self.hidden_dim), # (B, L, D) -> (B, L/4, 2*D)
        #     self.transfomer(width = 2*self.hidden_dim, layers=self.layers, heads = 2*self.hidden_dim//self.patch_size), # (B, L/4, 2*D) -> (B, L/4, 2*D)
        #     self.PatchMerging(input_length = 1/4*self.patch_length, dim = 2*self.hidden_dim), # (B, L/4, 2*D) -> (B, L/16, 4*D)
        #     self.transfomer(width = 4*self.hidden_dim, layers=self.layers, heads = 4*self.hidden_dim//self.patch_size), # (B, L/16, 4*D) -> (B, L/16, 4*D)
        #     self.PatchMerging(input_length = 1/16*self.patch_length, dim = 4*self.hidden_dim) # (B, L/16, 4*D) -> (B, L/64, 8*D)
        # )
        # self.bottleneck = nn.Sequential(
        #     self.transfomer(width = 8*self.hidden_dim, layers=self.layers, heads = 8*self.hidden_dim//self.patch_size), # (B, L/64, 8*D) -> (B, L/64, 8*D)
        #     self.transfomer(width = 8*self.hidden_dim, layers=self.layers, heads = 8*self.hidden_dim//self.patch_size) # (B, L/64, 8*D) -> (B, L/64, 8*D)
        # )
        # self.decoder = nn.Sequential(
        #     self.PatchExpanding(input_length = 1/64*self.patch_length, dim = 8*self.hidden_dim), # (B, L/64, 8*D) -> (B, L/16, 4*D)
        #     self.transfomer(width = 4*self.hidden_dim, layers=self.layers, heads = 4*self.hidden_dim//self.patch_size), # (B, L/16, 4*D) -> (B, L/16, 4*D)
        #     self.PatchExpanding(input_length = 1/16*self.patch_length, dim = 4*self.hidden_dim), # (B, L/16, 4*D) -> (B, L/4, 2*D)
        #     self.transfomer(width = 2*self.hidden_dim, layers=self.layers, heads = 2*self.hidden_dim//self.patch_size),    # (B, L/4, 2*D) -> (B, L/4, 2*D)
        #     self.PatchExpanding(input_length = 1/4*self.patch_length, dim = 2*self.hidden_dim), # (B, L/4, 2*D) -> (B, L, D)
        #     self.transfomer(width = self.hidden_dim, layers=self.layers, heads = self.hidden_dim//self.patch_size), # (B, L, D) -> (B, L, D)           (B,16*16,768)
        # ) 
        self.encoder = nn.Sequential(
            self.vit_model, # (B, C, H, W) -> (B, L, D) (B,16*16+1,768)
            self.Remove_class_token(), #(B, L+1, D) -> (B, L, D)
        )
        self.bottleneck = nn.Sequential(
            self.transfomer(width = self.hidden_dim, layers=self.layers, heads = self.hidden_dim//self.patch_size), # (B, L, D) -> (B, L, D)
            self.transfomer(width = self.hidden_dim, layers=self.layers, heads = self.hidden_dim//self.patch_size) # (B, L, D) -> (B, L, D)
        )
        self.decoder = nn.Sequential(
            self.transfomer(width = self.hidden_dim, layers=self.layers, heads = self.hidden_dim//self.patch_size), # (B, L, D) -> (B, L, D)
        )
        self.linear = Decoder(
            embed_dim = hidden_dim, 
            output_dim = output_dim, 
            patch_size= patch_size,
            output_size = output_size, 
            deprojection_type = deprojection_type, 
            with_bias = with_bias
        )

    def forward(self, x):
        x = self.encoder(x) # (B, C, H, W) -> (B, L/64, 8*D) (B,2*2,8*768)
        x = self.bottleneck(x) # (B, L/64, 8*D) -> (B, L/64, 8*D) (B, 2*2, 8*768)
        x = self.decoder(x) # (B, L/64, 8*D) -> (B, L, D) (B, 256, 768)
        x = self.linear(x) # (B, L, D) -> (B, C, H, W) (B, 1, 1024, 1024)
        return x
    
    def calculate_loss(self, image, image_2, weights, abs=False, criterion = nn.functional.mse_loss):
        recon = self(image)
        if abs:
            image_2 = torch.abs(image_2)
        mse = criterion(recon, image_2)
        mse_with_weight = weights*criterion(recon, image_2, reduction='none')
        mse_with_weight = mse_with_weight.mean()

        return mse_with_weight, mse
                 

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
            transformer_token_type = args.token_type,
            norm_type = args.norm_type
        )

def get_recon_model_from_args(args):
    return SolarReconModel(
            in_channels = args.in_channels,
            input_resolution = args.input_resolution,
            patch_size = args.patch_size,
            width = args.width,
            layers = args.layers,
            hidden_dim = args.hidden_dim,
            token_type = args.token_type,
            norm_type = args.norm_type,
            output_dim = args.output_dim,
            output_size = args.output_size,
            deprojection_type = args.deprojection_type,
            with_bias = args.with_bias
        )

def get_recon_unetlike_model_from_args(args):
    return SolarReconModel_Unet_like(
            in_channels = args.in_channels,
            input_resolution = args.input_resolution,
            patch_size = args.patch_size,
            width = args.width,
            vit_layers = args.vit_layers,
            transformer_layers= args.transformer_layers,
            hidden_dim = args.hidden_dim,
            token_type = args.token_type,
            norm_type = args.norm_type,
            output_dim = args.output_dim,
            output_size = args.output_size,
            deprojection_type = args.deprojection_type,
            with_bias = args.with_bias
        )


if __name__ =='__main__':
    net = Transformer(width = 1536, layers=12, heads = 24) # (B, L/4, 2*D) -> (B, L/4, 2*D)
    net.to('cuda:3')
    input()