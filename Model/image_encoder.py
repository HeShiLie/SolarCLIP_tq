import torch.nn as nn
import torch
from collections import OrderedDict



######image encoder  采用 ViT 结构

class LayerNorm(nn.LayerNorm):
   #使用的时候需要指定特征维度大小
   #处理float16数据 
   def forward(self, x: torch.Tensor) -> torch.Tensor:
      orig_tpye = x.dtype
      ret = super().forward(x.type(torch.float32))

      return ret.type(orig_tpye)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x) #GELU的近似计算，GELU是激活函数，用于神经网络的非线性变换


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([ #nn.sequential是一个有序的容器，模块将按照它们在构造函数中传递的顺序添加到其中
            ("c_fc", nn.Linear(d_model, d_model * 4)),  #orderdict是一个有序的字典，它的key是有序的
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
   def __init__(self, width: int, layers: int, heads: int, drop_out: float, attn_mask: torch.tensor = None):
      super().__init__()
      self.width = width
      self.layers = layers
      self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
      self.dropout = nn.Dropout(drop_out)

   def forward(self, x: torch.Tensor):
      
      return self.resblocks(x)

class VisionTransformer(nn.Module):
 def __init__(self, in_channels,input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
              token_type: str ):
    super().__init__()
    self.transformer_token_type = token_type
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)


    scale = width ** -0.5
    self.class_embedding = nn.Parameter(scale * torch.randn(width))
    self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
    self.ln_pre = LayerNorm(width)

    self.transformer = Transformer(width, layers, heads)

    self.ln_post = LayerNorm(width)
    self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

 def forward(self, x: torch.Tensor):
    x = self.conv1(x)
    
    x = x.reshape(x.shape[0],x.shape[1],-1)
    # print(x.shape)
    x = x.permute(0,2,1)

    x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)


    x = x.permute(1, 0, 2)  # NLD -> LND
    
    # print('transformer_input.shape:',x.shape)

    x = self.transformer(x)
    # print('transformer_output.shape:',x.shape)

    x = x.permute(1, 0, 2)  # LND -> NLD
    if self.transformer_token_type == 'class embedding':
        x = self.ln_post(x[:, 0,:]) #提取token的输出，即class_embedding的输出，即可表示全局特征
        # return [N, align_dim]
    elif self.transformer_token_type == 'all embedding':
        x = self.ln_post(x)
        # return [N, L, align_dim]
    else:
        pass

    # print('before embedding to joint space shape:',x.shape)

    if self.proj is not None:
        x = x @ self.proj   

    return x
 

# if __name__ == '__main__':
#     #mask = torch.empty(77, 77)
#     ##mask.fill_(float("-inf"))
#     #mask.triu_(1) 
#     model = VisionTransformer(input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512)
#     print(model(torch.randn(1, 3, 224, 224)).shape)
    
#     #print(mask.dtype)