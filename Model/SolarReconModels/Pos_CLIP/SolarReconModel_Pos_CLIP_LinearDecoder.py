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
    
    def calculate_loss(self, modal, modal_2, weights, abs, criterion = nn.functional.mse_loss):
        recon = self(modal)
        if abs:
            recon = torch.abs(recon)
        mse = criterion(recon, modal_2)
        mse_with_weight = weights*criterion(recon, modal_2, reduction='none') 
        mse_with_weight = mse_with_weight.mean()
        return mse_with_weight, mse

def get_SolarReconModel_from_args(args):
    Model = SolarReconModel(
        in_channels = args.in_channels, 
        input_resolution = args.input_resolution, 
        patch_size = args.patch_size, 
        width = args.width, 
        layers = args.vit_layers, 
        hidden_dim = args.hidden_dim, 
        token_type = args.token_type, 
        norm_type = args.norm_type,
        output_dim = args.output_dim,
        output_size = args.output_size,
        deprojection_type = args.deprojection_type,
        with_bias = args.with_bias)
    return Model