import torch.nn as nn
import torch
from .basic import Conv


class head(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            Conv(feat_dim, feat_dim // 2, k=1),
            Conv(feat_dim // 2, feat_dim, k=3, p=1),
            Conv(feat_dim, feat_dim // 2, k=1),
            Conv(feat_dim // 2, feat_dim, k=3, p=1),
        )
        self.pred = nn.Conv2d(feat_dim, 1 + self.num_classes + 4, 1)

    def init_bias(self):
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))
        nn.init.constant_(self.pred.bias[..., :1], bias_value)
        nn.init.constant_(self.pred.bias[..., 1 : 1 + self.num_classes], bias_value)

    def forward(self, x):
        return self.pred(self.head(x))
