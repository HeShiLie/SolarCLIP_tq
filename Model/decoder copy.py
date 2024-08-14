import torch.nn as nn
import torch
from einops import rearrange
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x) #GELU的近似计算

class LinearDecoder(nn.Module):
    def __init__(self, embed_dim, out_dim = 1024*4, hidden_dim_1=1024, hidden_dim_2=1024*2, hidden_dim_3=1024*3):
        super(LinearDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim #1024
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim_1),
            nn.ReLU(True),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(True),
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.ReLU(True),
            nn.Linear(hidden_dim_3, out_dim)
        )
            
    def forward(self, x: torch.Tensor):
        x = self.decoder(x)
        x = rearrange(x, 'b (h w) c -> b h w c', h = 16, w = 16)
        x = rearrange(x, 'b h w (p1 p2) -> b 1 (h p1) (w p2)', h = 16, w = 16, p1 = 64, p2 = 64)
        return x

def get_decoder_from_args(args):
    return LinearDecoder(
            embed_dim=args.embed_dim,
            out_dim=args.out_dim,
            hidden_dim_1=args.hidden_dim_1,
            hidden_dim_2=args.hidden_dim_2,
            hidden_dim_3=args.hidden_dim_3
        )
    
if __name__ == '__main__':
    model = LinearDecoder(512, 512)
    print(model)
    x = torch.randn(1, 768)
    print(model(x).shape)