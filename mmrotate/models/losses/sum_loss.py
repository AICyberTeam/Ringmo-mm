import torch
import torch.nn as nn
from ..builder import ROTATED_LOSSES


@ROTATED_LOSSES.register_module()
class SumLoss(nn.Module):
    def __init__(self):

        super(SumLoss, self).__init__()
        self.x = torch.randn(10)


    def forward(self, x):
        return torch.sum(x)