import torch
from torch import nn

def SelfAttention(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** 0.5
        self.qkv = nn.Linear(emb_dim, emb_dim * 3, bias=False)

        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.zeros_(self.qkv.bias)

    def forward(self, x):
        bs, n_locations, n_channels = x.shape
        q,k,v = self.qkv(x).view(bs, n_locations, 3, n_channels).unbind(dim=2)
        logits = torch.einsum('b i d, b j d -> b i j', q, k)
        logits *= self.scale
        attn = logits.softmax(-1)
        return torch.einsum('b i j, b j d -> b i d', attn, v)
