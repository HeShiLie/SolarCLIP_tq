import torch
import torch.nn as nn
import torch.nn.functional as F

from Modules.clip_modules.vit import VisionTransformer, Transformer, Remove_class_token
from Model.models import PatchMerging, PatchExpand, Decoder

"""
SolarReconModel_VAE Model.
This model use VAE architecture to reconstruct the solar image with encoder by clip model.
"""

class SolarReconModel_VAE_pos_CLIP(nn.Module):
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
                 with_bias: bool = True,
                 loss_type: str = 'MSE',
                 lambda_kl: float = 1e-1):
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
                
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.patch_length = (self.input_resolution // self.patch_size) ** 2
        self.hidden_dim = hidden_dim
        self.layers = transformer_layers
        self.loss_type = loss_type
        self.lambda_kl = lambda_kl


        self.encoder = nn.Sequential(
            self.vit_model, # (B, C, H, W) -> (B, L, D) (B,16*16+1,768)
            Remove_class_token(), #(B, L+1, D) -> (B, L, D)
            Transformer(width = self.hidden_dim, layers=self.layers, heads = self.hidden_dim//self.patch_size), # (B, L, D) -> (B, L, D)
        )
        self.fc_mu = nn.Linear(hidden_dim, hidden_dim)
        self.fc_var = nn.Linear(hidden_dim, hidden_dim)
        
        self.decoder = nn.Sequential(
            Transformer(width = self.hidden_dim, layers=self.layers, heads = self.hidden_dim//self.patch_size), # (B, L, D) -> (B, L, D)
            Transformer(width = self.hidden_dim, layers=self.layers, heads = self.hidden_dim//self.patch_size), # (B, L, D) -> (B, L, D)
            Decoder(embed_dim = hidden_dim, output_dim = output_dim, patch_size= patch_size,output_size = output_size, deprojection_type = deprojection_type, with_bias = with_bias)
        )
        
    def encode(self, x):
        x = self.encoder(x) # (B, C, H, W) -> (B, L, D) (B,16*16,768)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        log_var = torch.clamp(log_var, -10, 10)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def decode(self, z):
        x = self.decoder(z) 
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        x = self.decode(z)
        
        return x, mu, log_var
    
    def loss_function(self, recon_x, x, weights, mu, logvar):
        if self.loss_type == 'MSE':
            RECON_LOSS = F.mse_loss(recon_x, x, reduction='mean')
            RECON_LOSS_weighted = weights*F.mse_loss(recon_x, x, reduction='none')
            RECON_LOSS_weighted = RECON_LOSS_weighted.mean()
        elif self.loss_type == 'BCE':
            RECON_LOSS = F.binary_cross_entropy(recon_x, x, reduction='sum')
        else:
            raise ValueError(f"loss_type {self.loss_type} is not supported")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return RECON_LOSS_weighted + KLD*self.lambda_kl, RECON_LOSS, RECON_LOSS_weighted, KLD
    
    def sample(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z
    
def get_SolarReconModel_VAE_pos_CLIP_from_args(args):
    return SolarReconModel_VAE_pos_CLIP(
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