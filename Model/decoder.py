import torch.nn as nn
import torch
from einops import rearrange
class LinearDecoder(nn.Module):
    def __init__(self, image_size = 1024, patch_size = 64, embed_dim = 768):
        super(LinearDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_h = image_size // patch_size
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2, bias = False),
            nn.SiLU(),
            nn.Linear(embed_dim*2, embed_dim*4, bias = False),
            nn.SiLU(),
            nn.Linear(embed_dim*4, patch_size**2, bias = False)
        )
            
    def forward(self, x: torch.Tensor):
        # b np d ->b np (h w) 
        x = self.decoder(x)
        x = rearrange(x, 'b (h w) c -> b h w c', h = self.n_h, w = self.n_h)
        x = rearrange(x, 'b h w (p1 p2) -> b 1 (h p1) (w p2)', h = self.n_h, w = self.n_h, p1 = self.patch_size, p2 = self.patch_size)
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