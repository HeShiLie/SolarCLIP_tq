from collections import OrderedDict

import torch
import torch.nn as nn

from einops import rearrange

from Modules.clip_modules.vit import VisionTransformer

class patch_norm(nn.Module):
    def __init__(self, d_model = 768, norm_type = 'bn1d', eps = 1e-5):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == 'bn1d':
            self.norm = nn.BatchNorm1d(d_model, eps)
        elif norm_type == 'ln':
            self.norm = nn.LayerNorm(d_model, eps)
        else:
            raise ValueError('norm_type should be bn1d or ln')

    def forward(self, x):
        # x.size: (b, num_patches, d_model) (b (n_t n_h n_w) d)
        # x shape: (batch_size, num_frames, num_patches, d_model)
        if self.norm_type == 'bn1d':
            x = rearrange(x, 'b p d -> b d p')
            x = self.norm(x)
            x = rearrange(x, 'b d p -> b p d')
        elif self.norm_type == 'ln':
            x = self.norm(x)
        else:
            raise ValueError('norm_type should be bn1d or ln')
        return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x) 

    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, norm_type: str = 'bn1d'):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = patch_norm(d_model = d_model, norm_type = norm_type)
        self.mlp = nn.Sequential(OrderedDict([ #nn.sequential是一个有序的容器，模块将按照它们在构造函数中传递的顺序添加到其中
            ("c_fc", nn.Linear(d_model, d_model * 4)),  #orderdict是一个有序的字典，它的key是有序的
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = patch_norm(d_model = d_model, norm_type = norm_type)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
   def __init__(self, width: int, layers: int, heads: int, drop_out: float=0.0, attn_mask: torch.tensor = None):
      super().__init__()
      self.width = width
      self.layers = layers
      self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
      self.dropout = nn.Dropout(drop_out)

   def forward(self, x: torch.Tensor):
      
      return self.resblocks(x)
    
class Decoder(nn.Module):
    def __init__(self, 
                 embed_dim: int = 768,
                 output_dim: int = 1,
                 patch_size: int = 64,
                 output_size: int = 1024,
                 nonlinear: str = 'gelu',
                 deprojection_type: str = 'linear',
                 with_bias: bool = True):
        super().__init__()
        if output_size % patch_size != 0:
            raise ValueError(f"Image size {output_size} must be divisible by patch size {patch_size}, now is not divisible.")
        self.n_h = output_size // patch_size
        self.h = patch_size
        self.c = output_dim
        self.deprojection_type = deprojection_type
        self.linear = nn.Linear(embed_dim, embed_dim, bias=with_bias)
        if nonlinear == 'silu':
            self.nonlinear = nn.SiLU()
        elif nonlinear == 'gelu':
            self.nonlinear = QuickGELU()
        if deprojection_type == 'linear':
            self.deprojection = nn.Sequential(
                nn.Linear(embed_dim, embed_dim*2, bias=with_bias),
                self.nonlinear,
                nn.Linear(embed_dim*2, embed_dim*4, bias=with_bias),
                self.nonlinear,
                nn.Linear(embed_dim*4, output_dim*patch_size**2, bias=with_bias)
            )
        elif deprojection_type == 'conv2d':
            self.deprojection = nn.ConvTranspose2d(embed_dim, output_dim, kernel_size=patch_size, stride=patch_size, bias=with_bias)
        else:
            raise ValueError(f"{deprojection_type} deprojection is not supported")
        
    def forward(self, x):
        x = self.linear(x) # B p d
        x = self.nonlinear(x)
        x = self.deprojection(x)
        # if linear, x = [B p (c*h*w)]
        # elif conv2d, x = [B p c h w]
        if self.deprojection_type == 'linear':
            x = rearrange(x, 'b (n_h n_w) (c h w) -> b c (n_h h) (n_w w)', n_h=self.n_h, n_w=self.n_h, c=self.c, h=self.h, w=self.h)
        elif self.deprojection_type == 'conv2d':
            x = rearrange(x, 'b (n_h n_w) c h w -> b c (n_h h) (n_w w)', n_h=self.n_h, n_w=self.n_h, c=self.c, h=self.h, w=self.h)
        return x

class Remove_class_token(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x[:, 1:, :]

class SolarReconModel_Pos_CLIP_transformerDecoder(nn.Module):
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
        # self.PatchMerging = PatchMerging
        # self.PatchExpanding = PatchExpand
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

def get_SolarReconModel_Pos_CLIP_transformerDecoder_from_args(args):
    Model = SolarReconModel_Pos_CLIP_transformerDecoder(
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
        with_bias = args.with_bias)
    return Model