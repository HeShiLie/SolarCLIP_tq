from torch import nn
import numpy as np
import torch
from collections import OrderedDict
from einops import rearrange

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
        self.gelu = QuickGELU()

        if deprojection_type == 'linear':
            self.deprojection = nn.Sequential(
                nn.Linear(embed_dim, embed_dim*2, bias=with_bias),
                QuickGELU(),
                nn.Linear(embed_dim*2, embed_dim*4, bias=with_bias),
                QuickGELU(),
                nn.Linear(embed_dim*4, output_dim*patch_size**2, bias=with_bias)
            )
        elif deprojection_type == 'conv2d':
            self.deprojection = nn.ConvTranspose2d(embed_dim, output_dim, kernel_size=patch_size, stride=patch_size, bias=with_bias)
        else:
            raise ValueError(f"{deprojection_type} deprojection is not supported")
        
    def forward(self, x):
        x = self.linear(x) # B p d
        x = self.gelu(x)
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