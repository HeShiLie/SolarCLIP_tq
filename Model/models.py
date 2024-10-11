from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import torch
from collections import OrderedDict
from einops import rearrange
import math

class LayerNorm(nn.LayerNorm):
   #使用的时候需要指定特征维度大小
   #处理float16数据 
   def forward(self, x: torch.Tensor) -> torch.Tensor:
      orig_tpye = x.dtype
      ret = super().forward(x.type(torch.float32))

      return ret.type(orig_tpye)
   
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
        return x * torch.sigmoid(1.702 * x) #GELU的近似计算，GELU是激活函数，用于神经网络的非线性变换
    
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_length (tuple[int]): Number of input features.  
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_length, dim, norm_layer=patch_norm):
        super().__init__()
        self.input_length = input_length
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        H, W = int(np.sqrt(self.input_length)), int(np.sqrt(self.input_length))
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B (H/2*W/2) 4*C 
        # 可以换成rearange的形式吗

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_length
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchExpand(nn.Module):
    def __init__(self, input_length, dim, dim_scale=2, norm_layer=patch_norm):
        super().__init__()
        self.input_length = input_length
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = int(np.sqrt(self.input_length)), int(np.sqrt(self.input_length))
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)  # B (H*2) (W*2) C/2
        x= self.norm(x)

        return x

class Remove_class_token(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x[:, 1:, :]
    
class Encoder(nn.Module):
    def __init__(self, 
                 input_size: int = 1024,
                 embed_dim: int = 768,
                 input_dim: int = 1,
                 patch_size: int = 64,
                 dropout_prob: float = 0.1):
        super().__init__()
        if input_size % patch_size != 0:
            raise ValueError(f"Image size {input_size} must be divisible by patch size {patch_size}, now is not divisible.")
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.ln_pre = patch_norm(d_model = embed_dim, norm_type = 'bn1d')
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x) # (B,C,H,W) -> (B,D,H/patch_size,W/patch_size)
        x = x.reshape(x.shape[0], x.shape[1], -1) # (B, D, H/patch_size, W/patch_size) -> (B, D, L=H/patch_size*W/patch_size)
        x = x.permute(0, 2, 1) # (B, D, L) -> (B, L, D)
        x = self.ln_pre(x) # (B, L, D) -> (B, L, D)
        x = self.dropout(x) # (B, L, D) -> (B, L, D)
        return x
    
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

class PretrainModel(nn.Module):
    def __init__(self, 
                input_size: int = 1024,
                embedding_dim: int = 768,
                input_dim: int = 1,
                patch_size: int = 64,
                dropout_prob: float = 0.1,
                output_dim: int = 1,
                output_size: int = 1024,
                deprojection_type: str = 'linear',
                with_bias: bool = True):
        super().__init__()
        self.encoder = Encoder(input_size=input_size, embed_dim=embedding_dim, input_dim=input_dim, patch_size=patch_size, dropout_prob=dropout_prob)
        self.decoder = Decoder(embed_dim=embedding_dim, output_dim=output_dim, patch_size=patch_size, output_size=output_size, deprojection_type=deprojection_type, with_bias=with_bias)

    def forward(self, x):
        x = self.encoder(x) # (B, C, H, W) -> (B, L, D)
        x = self.decoder(x) # (B, L, D) -> (B, C, H, W)
        return x
    
    def calculate_loss(self, image, weights, criterion = torch.nn.functional.mse_loss):
        # batch: [batch, modal, channel, height, width]
        # image from batch: [batch, channel, height, width]  
        recon = self(image)
        mse = criterion(recon, image)
        mse_with_weight = weights*criterion(recon, image, reduction='none') 
        mse_with_weight = mse_with_weight.mean()
        return mse_with_weight, mse
    
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

class VisionTransformer(nn.Module):
 def __init__(self, in_channels, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
              token_type: str , norm_type: str = 'bn1d'):
    super().__init__()
    self.transformer_token_type = token_type
    # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
    self.conv1 = Encoder(input_dim=in_channels, embed_dim=width, input_size=input_resolution, patch_size=patch_size, dropout_prob=0.1)

    scale = width ** -0.5
    self.class_embedding = nn.Parameter(scale * torch.randn(width))
    self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
    self.ln_pre = patch_norm(width, norm_type)

    self.transformer = Transformer(width, layers, heads)

    self.ln_post = patch_norm(width, norm_type)
    self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

 def forward(self, x: torch.Tensor):
    x = self.conv1(x)
    
    # x = x.reshape(x.shape[0],x.shape[1],-1)
    # print(x.shape)
    # x = x.permute(0,2,1) # (B, D, L) -> (B, L, D)

    x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)

    #! donot permute here, we focus on the global feature
    # x = x.permute(1, 0, 2)  # BLD -> LBD
    x = self.transformer(x)
    # x = x.permute(1, 0, 2)  # LBD -> BLD

    if self.transformer_token_type == 'class embedding':
        x = self.ln_post(x[:, 0,:]) #提取token的输出，即class_embedding的输出，即可表示全局特征
        # return [N, align_dim]
    elif self.transformer_token_type == 'all embedding':
        x = self.ln_post(x)
        # return [N, L, align_dim]

    if self.proj is not None:
        x = x @ self.proj   

    return x

 def get_last_selfattention(self, x):
    x = self.conv1(x)
    
    x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)

    x = self.transformer(x)

    return x
 
 # ------------------ VAE ------------------
class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias = True, out_proj_bias= True):
        super().__init__()
        
        #Combining the Wq, Wk, and Wv matrices into one
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias= in_proj_bias)
        
        #Represent the Wo Matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias= out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x, causal_mask = False):
        
        input_shape = x.shape
        
        batch_size, sequence_length, d_embed = input_shape
        
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k ,v = self.in_proj(x).chunk(3, dim= -1)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)
        
        weight = q @ k.transpose(-1,-2)
        
        if causal_mask:     
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim = -1)
        
        output = weight @ v
        
        output = output.transpose(1,2)
        
        output = output.reshape(input_shape)
        
        output = self.out_proj(output)
        
        return output


class VAE_AttentionBlock(nn.Module):
    def __init__(self, 
                 channels, 
                 num_groups: int = 32):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups, channels)
        self.attention = SelfAttention(1,channels)
        
    def forward(self,x):
        
        residue = x
        x = self.groupnorm(x)
        
        n, c, h, w = x.shape
        x = x.view(n ,c,h *w)
        x = x.permute(0,2,1)                
        x = self.attention(x)
        x = x.permute(0,2,1)        
        x = x.view(n,c,h,w)
    
        x += residue
        return x
    
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups: int = 32,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.groupnorm_pre = nn.GroupNorm(num_groups, out_channels//2)
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.groupnorm_post = nn.GroupNorm(num_groups, out_channels)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.nonlinear = nn.ELU()
            
    def forward(self,x):
        residue = self.residual_layer(x)
        x = self.nonlinear(self.groupnorm_pre(self.conv1(x)))
        x = self.conv2(x)
        return self.nonlinear(self.groupnorm_post(x + residue))
class VAE(nn.Module):
    def __init__(self, 
                 input_size: int = 1024, 
                 image_channels: int = 1, 
                 hidden_dim: int = 64,
                 group_nums: int = 16,
                 latent_dim: int = 3,
                 loss_type: str = 'MSE',
                 lambda_kl: float = 1.0):
        super().__init__()
        self.input_size = input_size
        self.image_channels = image_channels
        self.group_nums = group_nums
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        self.lambda_kl = lambda_kl

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, kernel_size=3, stride=1, padding=1),  # B, 1, 1024, 1024 -> B, 64, 1024, 1024
            nn.ELU(),
            VAE_ResidualBlock(hidden_dim, hidden_dim*2, kernel_size=4, stride=4, padding=0),  # B, 64, 1024, 1024 -> B, 128, 256, 256
            VAE_ResidualBlock(hidden_dim*2, hidden_dim*4, kernel_size=4, stride=4, padding=0),  # B, 128, 256, 256 -> B, 256, 64, 64
            # VAE_AttentionBlock(hidden_dim*8),  # B, 256, 16, 16 -> B, 256, 16, 16
            nn.GroupNorm(group_nums, hidden_dim*4), # B, 256, 64, 64 -> B, 256, 64, 64
            nn.ELU(),
            nn.Conv2d(hidden_dim*4, self.latent_dim*2, kernel_size=1, stride=1, padding=0), # B, 256, 64, 64 -> B, 6, 64, 64
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, hidden_dim*4, kernel_size=3, stride=1, padding=1), # B, 3, 64, 64 -> B, 256, 64, 64
            nn.ELU(),
            # VAE_AttentionBlock(hidden_dim*8), # B, 256, 16, 16 -> B, 256, 16, 16

            nn.Upsample(scale_factor=4, mode='nearest'), # B, 256, 64, 64 -> B, 256, 256, 256
            VAE_ResidualBlock(hidden_dim*4, hidden_dim*2, kernel_size=3, stride=1, padding=1), # B, 256, 256, 256 -> B, 128, 256, 256
            
            nn.Upsample(scale_factor=4, mode='nearest'), # B, 128, 256, 256 -> B, 128, 1024, 1024
            VAE_ResidualBlock(hidden_dim*2, hidden_dim, kernel_size=3, stride=1, padding=1), # B, 128, 1024, 1024 -> B, 64, 1024, 1024

            nn.GroupNorm(group_nums, hidden_dim), # B, 64, 1024, 1024 -> B, 64, 1024, 1024
            nn.ELU(),
            nn.Conv2d(hidden_dim, image_channels, kernel_size=3, stride=1, padding=1), # B, 64, 1024, 1024 -> B, 1, 1024, 1024
        )

    def encode(self, x):
        """
        x: (B, C, H, W) eg: (B, 1, 1024, 1024)
        output: (B, C_out, H_out, W_out) eg: (B, 3, 64, 64)
        """
        x = self.encoder(x) # (B, C, H, W) -> (B, C_out, H_out, W_out)
        mu, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30, 30)
        return mu, logvar

    def reparameterize(self, mu, logvar, scale=1.0):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        z = z*scale
        return z

    def decode(self, z):
        """
        z: (B, latent_dim, H_out, W_out) eg: (B, 3, 16, 16)
        output: (B, input_dim, H, W) eg: (B, 1, 1024, 1024)
        """
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        # print(f"After encoder: {torch.cuda.memory_allocated()/1e6} MB")
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
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

def get_VAE_model_from_args(args):
    return VAE(
        input_size=args.input_size,
        image_channels=args.image_channels,
        hidden_dim=args.hidden_dim,
        group_nums=args.num_groups,
        latent_dim=args.latent_dim,
        loss_type=args.loss_type,
        lambda_kl=args.lambda_ratio
    )
 # ------------------ DiT ------------------

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size: int=768, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            QuickGELU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (B, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None] # (B, 1) * (1, D/2) -> (B, D/2)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size) # (B, 1) -> (B, frequency_embedding_size)
        t_emb = self.mlp(t_freq) # (B, frequency_embedding_size) -> (B, D)
        return t_emb
class LabelEmbedder(nn.Module):  # todo
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class DynamicScaledAttention(nn.Module):
    def __init__(self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,):
        super(DynamicScaledAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scale_qkv_layer = nn.Linear(dim, 3, bias=qkv_bias)
        self.fused_attn = False
        
    def forward(self, x, y):
        B, L, D = x.shape
        assert x.shape[:2] == y.shape[:2], "x and y must have the same batch size and sequence length"
        qkv = self.qkv(x).reshape(B, L, 3, self.dim).permute(2, 0, 1, 3) # (3, B, L, D)
        scale = self.scale_qkv_layer(y).permute(2, 0, 1).unsqueeze(-1)  # (3, B, L, 1)
        scale_qkv = (scale * qkv).reshape(3, B, L, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)  # (3, B, H, L, D)
        q, k, v = scale_qkv.unbind(dim=0)  
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, L, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning. 自适应 层归一化
    """
    def __init__(self, width: int = 768, n_head: int = 16, norm_type: str = 'bn1d'):
        super().__init__()
        self.norm1 = patch_norm(width, norm_type= norm_type)
        self.dynamic_scaled_attn = DynamicScaledAttention(dim=width, num_heads=n_head)
        self.mlp = nn.Sequential(OrderedDict([ 
            ("c_fc", nn.Linear(width, width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(width * 4, width))
        ]))
        self.norm2 = patch_norm(width, norm_type= norm_type)
        self.adaLN_modulation = nn.Sequential(
            QuickGELU(),
            nn.Linear(width, 6 * width, bias=True)
        )

    def forward(self, x, c_y, c_t): # c may be timestep embedding + clip embedding (B, 1, D)+ (B, L, D)
        """
        x: (B, L, D) # input tensor
        c_y: (B, L, D) # conditioning vector of CLIP embedding
        c_t: (B, 1, D) # conditioning vector of timestep embedding
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c_t).chunk(6, dim=2)
        x = x + gate_msa * self.dynamic_scaled_attn(modulate(self.norm1(x), shift_msa, scale_msa), c_y) # (B,L,D) -> (B,L,D)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)) # (B,L,D) -> (B,L,D)
        return x
class DiT(nn.Module):  # to verify
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=64,
        patch_size=4,
        in_channels=3,
        width=768,
        norm_type='bn1d',
        depth=28,
        num_heads=16,
        token_type='all embedding',
        layers=12,
        learn_sigma=True,
        deprojection_type='linear',
        with_bias=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.output_size = self.input_size = input_size
        self.patch_size = patch_size
        self.width = width
        self.token_type = token_type
        self.norm_type = norm_type
        self.num_heads = num_heads
        self.layers = layers

        self.x_embedder = Encoder(input_dim=self.in_channels, embed_dim=width, input_size=self.input_size, patch_size=self.patch_size, dropout_prob=0.1)
        self.t_embedder = TimestepEmbedder(width)
        self.y_vit = VisionTransformer(
            in_channels=1,
            input_resolution=1024,
            patch_size=64,
            width=768,
            layers=12,
            heads=12,
            output_dim=768,
            token_type=self.token_type,
            norm_type='bn1d',
        )
        self.Remove_class_token = Remove_class_token
        self.y_embedder = nn.Sequential(
            self.y_vit,
            self.Remove_class_token(), #(B, L+1, D) -> (B, L, D)
        )
        self.num_patches = (input_size // patch_size )** 2
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, width), requires_grad=False)
        self.blocks = nn.ModuleList([
            DiTBlock(width, num_heads, norm_type) for _ in range(depth)
        ])
        self.final_decoder = Decoder(width, self.out_channels, patch_size, self.output_size, 'silu', deprojection_type, with_bias)
        self.initialize_weights()

    def initialize_weights(self):
        pass
        # Initialize transformer layers:
        # def _basic_init(module):
        #     if isinstance(module, nn.Linear):
        #         torch.nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)
        # self.apply(_basic_init)

        # # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table: # 这里用clip的embedding初始化
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        # nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # # Zero-out adaLN modulation layers in DiT blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # # Zero-out output layers:
        # nn.init.constant_(self.final_decoder.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_decoder.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.final_decoder.linear.weight, 0)
        # nn.init.constant_(self.final_decoder.linear.bias, 0)

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images) eg: (B, 3, 64, 64)
        t: (B,) tensor of diffusion timesteps
        y: (B, L, D) tensor of token generated by CLIP model
        """
        x = self.x_embedder(x) + self.pos_embed  # (B, C, H, W) -> (B, L, D)
        t = self.t_embedder(t).unsqueeze(1)                   # (B, 1, D)
        for block in self.blocks:
            x = block(x, y, t)                   # (B, L, D)
        x = self.final_decoder(x)                # (B, L, D) -> (B, C*2, H, W)
        return x

def get_DiT_model_from_args(args):
    """
    Create a DiT model from a dictionary of arguments.
    """
    return DiT(
        input_size=args.input_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        width=args.width,
        norm_type=args.norm_type,
        depth=args.depth,
        num_heads=args.num_heads,
        token_type=args.token_type,
        layers=args.layers,
        learn_sigma=args.learn_sigma,
        deprojection_type=args.deprojection_type,
        with_bias=args.with_bias,
    )

# ------------------ U-Net ------------------


# ------------------ U-ViT ------------------