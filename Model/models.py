from torch import nn
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
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
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
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, width: int = 768, n_head: int = 16, norm_type: str = 'bn1d'):
        super().__init__()
        self.norm1 = patch_norm(width, norm_type= norm_type)
        self.attn = nn.MultiheadAttention(width, n_head)
        self.mlp = nn.Sequential(OrderedDict([ 
            ("c_fc", nn.Linear(width, width * 4))
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(width * 4, width))
        ]))
        self.norm2 = patch_norm(width, norm_type= norm_type)
        self.adaLN_modulation = nn.Sequential(
            QuickGELU(),
            nn.Linear(width, 6 * width, bias=True)
        )

    def forward(self, x, c):
        """
        x: (B, L, D)
        c: (B, 1, D)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa)) # (B,L,D) -> (B,L,D)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)) # (B,L,D) -> (B,L,D)
        return x
    
class DiTDecoder(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        patch_size: int=64,
        in_channels: int=4,
        width: int=768,
        depth: int=28,
        num_heads: int=16,
        output_dim: int=1,
        output_size: int=1024,
        deprojection_type: str='linear',
        with_bias: bool=True,
        norm_type='bn1d',
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.t_embedder = TimestepEmbedder(width)
        self.y_embedder = LabelEmbedder(num_classes, width, class_dropout_prob)

        self.blocks = nn.ModuleList([
            DiTBlock(width, num_heads, norm_type) for _ in range(depth)
        ])
        self.final_decoder = Decoder(width, output_dim, patch_size, output_size, deprojection_type, with_bias)
        
        self.initialize_weights()

    def initialize_weights(self): # todo
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:  # todo
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (B, L, D) tensor of input features
        t: (N,) tensor of diffusion timesteps
        y: (B, 1, D) tensor of class labels
        """
        t = self.t_embedder(t)                   # (B, D)
        y = self.y_embedder(y, self.training)    # (B, 1, D)
        c = t.unsqueeze(dim =1) + y              # (B, 1, D) -> (B, 1, D)
        for block in self.blocks:
            x = block(x, c)                      # (B, L, D) -> (B, L, D)
        x = self.final_decoder(x)                # (B, L, D) -> (B, C, H, W)
        return x
