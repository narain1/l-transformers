import torch
from torch import nn
from collections import OrderedDict
from torch.nn import init
import torch.nn.functional as F

class ECAAttention(nn.Module):
    def __init__(self, ks=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=ks, padding=(ks-1)//2)

    def forward(self, x):
        o = F.adaptive_avg_pool2d(x, 1)
        o = o.squeeze(-1).permute(0,2,1)
        o = self.conv(o).sigmoid()
        o = o.permute(0,2,1).unsqueeze(-1)
        return x*o.expand_as(x)
