import torch
from torch import nn


class MHA(nn.Module):
    def __init__(self,
                 n_channels,
                 n_heads,
                 drop_attn_p=0.0,
                 drop_out_p=0.0,
                 block_size=128,
                 causal=False
                 ):
        super().__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.scale = (n_channels // n_heads) ** -0.5
        self.qkv = nn.Linear(n_channels, 3 * n_channels)
        if causal:
            self.register_buffer(
                "mask", torch.tril(torch.ones(1, 1, block_size, block_size)).log()
            )
        else:
            self.mask = None
        if drop_attn_p > 0.0:
            self.dropout_attn = nn.Dropout(drop_attn_p)
        else: self.dropout_attn = None
        if drop_out_p > 0.0:
            self.dropout_out = nn.Dropout(drop_out_p)
        else: self.dropout_out = None

        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.zeros_(self.qkv.bias)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        bs, n, n_channels = x.shape
        n_heads = self.n_heads
        q,k,v = self.qkv(x).view(bs, n, 3, n_heads, -1).unbind(2)
        logits = torch.einsum('bthc,bshc->bhts', q, k)
        logits *= self.scale
        if self.mask is not None:
            logits += self.mask
        attn = logits.softmax(-1)
        if self.dropout_attn is not None:
            attn = self.dropout_attn(attn)
        output = torch.einsum('bhts,bshc->bthc', attn, v)
        output = output.reshape(bs, n, n_channels)
        output = self.proj(output)
        if self.dropout_out is not None:
            output = self.dropout_out(output)
        return output
