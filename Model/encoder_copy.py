from torch import nn
import torch

from .TQ_image_encoder import patch_norm,Transformer
    
class Embeddings(nn.Module):
    def __init__(self, width: int, input_resolution: int, patch_size: int, in_channels: int, hidden_dropout_prob: float):
        super().__init__()
        self.scale = width ** -0.5
        self.class_embedding = nn.Parameter(self.scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(self.scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.ln_pre = patch_norm('bn1d', width)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)   # (B, C, H, W) -> (B, D, H//kernel_size, W//kernel_size)
        x = x.reshape(x.shape[0],x.shape[1],-1)        # (B, D, H0, W0) -> (B, D, L=H0*W0)
        x = x.permute(0,2,1)        # (B, D, L) -> (B, L, D)
        x = self.ln_pre(x)      # (B, L, D) -> (B, L, D)

        # (B, L, D) -> (B, L+1, D)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  
        x = x + self.positional_embedding.to(x.dtype)
        x= self.dropout(x)

        # return = (B, L+1, D)
        return x 
    
class ViTransformer(nn.Module):
    def __init__(self, 
                 input_resolution: int, patch_size: int, in_channels: int, hidden_dropout_prob_embeddings: float,
                 width: int, layers: int, heads: int, hidden_dropout_prob_transformer: float,
                 token_type: str): 
        super().__init__()
        self.transformer_token_type = token_type
        self.Embeddings = Embeddings(width,input_resolution,patch_size,in_channels, hidden_dropout_prob_embeddings)
        self.transformer = Transformer(width, layers, heads, hidden_dropout_prob_transformer)
        self.ln_post = patch_norm('bn1d', width)
        
    def forward(self, x: torch.Tensor):

        # (B, C, H, W) -> (B, L+1, D)
        hidden_state = self.Embeddings(x)
        # (B, L+1, D) -> (B, L+1, D)
        features = self.transformer(hidden_state)
        if self.transformer_token_type == 'class embedding':
            features = self.ln_post(features[:, 0,:])
        elif self.transformer_token_type == 'all embedding':
            features = self.ln_post(features)
        else:
            raise ValueError('transformer_token_type should be either class embedding or all embedding')
        
        # return = (B, D) or (B, L, D)
        return features

        
class PretrainModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = x[:,1:,:]
        x = self.decoder(x)
        return x
            