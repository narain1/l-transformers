from torch import nn
import torch
import torch.nn.functional as F

class SEAttention(nn.Module):
    def __init__(self, c_in, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in//reduction, c_in, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, *_ = x.size()
        o = F.adaptive_avg_pool2d(x, 1).view(b, c)
        o = self.fc(o).view(b, c, 1, 1)
        return x * o.expand_ax(x)
