import torch
import torch.nn as nn
from .basic import Conv


class SPP(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.pool1 = nn.MaxPool2d(5, 1, 2)
        self.pool2 = nn.MaxPool2d(9, 1, 4)
        self.pool3 = nn.MaxPool2d(13, 1, 6)

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        y = torch.cat([x, x1, x2, x3], dim=1)
        return y


class neck(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.layer = nn.Sequential(SPP(), Conv(feat_dim * 4, feat_dim, k=1))

    def forward(self, x):
        return self.layer(x)
