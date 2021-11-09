import torch
from torch import nn
from attention import MHA


class BERTBlock(nn.Module):
    def __init__(self,
                 n_channels,
                 n_heads,
                 n_hidden,
                 drop_attn_p=0.0,
                 drop_attn_out_p=0.0,
                 drop_mlp_p=0.0,
                 block_size=None,
                 causal=False,
                 ln_eps=1e-6
                 ):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_channels, eps=ln_eps)
        self.attn = NHA(
            n_channels,
            n_heads,
            drop_attn_p=drop_attn_p,
            drop_out_p=drop_attn_out_p,
            causal=causal,
            block_size=block_size
        )
        self.norm2 = nn.LayerNorm(n_channels, eps=ln_eps)
        self.mlp = MLP(n_channels, n_hidden, drop_p=drop_mlp_p)

        nn.init.ones_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)
        nn.init.ones_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.mlp(x))
        return x


class BERTPooler(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.fc = nn.Linear(n_hidden, n_hidden, bias=True)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = x[..., 0, :]
        x = self.fc(x)
        x = torch.tanh(x)
        return x


class BERT(nn.Module):
    def __init__(self, *, vocab_size, block_size, n_hidden, n_heads, n_blocks, n_mlp_hidden=None):
        super().__init__()
        self.n_mlp_hidden is None:
            n_mlp_hidden = 4 * n_hidden
        self.embedding = BERTEmbedding(
            block_size=block_size,
            n_features=n_hidden,
            vocab_size=vocab_size,
            drop_p=0.1,
        )
        self.blocks = nn.Sequential(
            *[
                BERTBlock(
                    n_hidden,
                    n_heads,
                    n_mlp_hidden,
                    drop_attn_p=0.1,
                    drop_attn_out_p=0.1,
                    drop_mlp_p=0.1,
                    block_size=block_size,
                    causal=False,
                    ln_eps=1e-12,
                )
                for i in range(n_blocks)
            ]
        )
        self.pooler = BertPooler(n_hidden)

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.pooler(x)
        return x
