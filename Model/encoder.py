from torch import nn
import torch

from Model.get_weights import get_weights
from .TQ_image_encoder import patch_norm,Transformer
    
class Embeddings(nn.Module):
    def __init__(self, width: int, input_resolution: int, patch_size: int, in_channels: int, hidden_dropout_prob: float):
        super().__init__()
        self.scale = width ** -0.5
        self.class_embedding = nn.Parameter(self.scale * torch.randn(width))
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.ln_pre = patch_norm('bn1d', width)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x: torch.Tensor):
        #x shape = [batch_size, num_channels, height, width]
        x = self.conv1(x)
        x = x.reshape(x.shape[0],x.shape[1],-1)
        #x shape = [batch_size, num_patches, embed_dim]
        x = x.permute(0,2,1)
        x = self.ln_pre(x)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        #x shape = [batch_size, num_patches, embed_dim]
        x= self.dropout(x)
        # output shape = [batch_size, num_patches, embed_dim=width]
        return x
    
# class ViTransformer(nn.Module):
#     def __init__(self, width: int, layers: int, heads: int, hidden_dropout_prob: float, output_dim: int):
#         super().__init__()
#         self.transformer = Transformer(width, layers, heads, hidden_dropout_prob)
#         self.ln_post = patch_norm('bn1d', width)
#         self.proj = nn.Parameter(torch.randn(width, output_dim) * (width ** -0.5))
#         self.Embeddings = Embeddings(width,input_resolution,patch_size,in_channels, hidden_dropout_prob)

#     def forward(self, x: torch.Tensor):
#         hidden_state = self.Embeddings(x)
#         features = self.transformer(hidden_state)
#         if self.transformer_token_type == 'class embedding':
#             features = self.ln_post(features[:, 0,:])
#         elif self.transformer_token_type == 'all embedding':
#             features = self.ln_post(features)
#         else:
#             raise ValueError('transformer_token_type should be either class embedding or all embedding')

        
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
    
    def calculate_loss(self, image, criterion = torch.nn.functional.mse_loss, weights_type = '3sgm-discrete'):
    # [batch, channel, height, width]  batch:[batch, modal, channel, height, width]
    
        weights, _ = get_weights(weights_type, image)
        feature = self.encoder(image)
        feature = feature[:,1:,:]  ##all embedding
        recon = self.decoder(feature)
        loss_mse = criterion(recon, image)
        loss = weights*criterion(recon, image, reduction='none') 
        # loss = torch.sum(loss).item()
        loss = loss.mean()
        return loss, loss_mse
            