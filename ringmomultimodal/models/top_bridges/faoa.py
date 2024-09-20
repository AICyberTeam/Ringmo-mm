from collections import OrderedDict
import torch
from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F
from ringmomultimodal.models.builder import BRIDGE
from .base import TopBridgeBase
from debug_tools import show_value


@BRIDGE.register_module()
class FAOATopBridge(TopBridgeBase):
    def __init__(self, hidden_dim=256, dropout=0.1, leaky=False, coordmap=True, language_channel=768,
                 vision_channels=(1024, 512, 256)):
        super(FAOATopBridge, self).__init__()
        self.coordmap = coordmap
        self.mapping_visu = nn.Sequential(OrderedDict([
            ('0', ConvBatchNormReLU(vision_channels[0], hidden_dim, 1, 1, 0, 1, leaky=leaky)),
            ('1', ConvBatchNormReLU(vision_channels[1], hidden_dim, 1, 1, 0, 1, leaky=leaky)),
            ('2', ConvBatchNormReLU(vision_channels[2], hidden_dim, 1, 1, 0, 1, leaky=leaky))
        ]))
        self.mapping_lang = torch.nn.Sequential(
            nn.Linear(language_channel, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        embin_size = hidden_dim * 2
        if self.coordmap:
            embin_size += 8

    def forward(self, fv, fl, **kwargs):

        fv_norm = []
        for i, f in enumerate(fv):
            fv_norm.append(self.mapping_visu._modules[str(i)](fv[i]))
            fv_norm[i] = F.normalize(fv_norm[i], p=2, dim=1)

        if isinstance(fl, list) and len(fl) == 1:
            fl = fl[0]
        if isinstance(fl, tuple):
            fl, word_feature = fl
        fl = (fl[-1][:, 0, :] + fl[-2][:, 0, :] \
              + fl[-3][:, 0, :] + fl[-4][:, 0, :]) / 4
        fl = self.mapping_lang(fl)
        fl_norm = F.normalize(fl, p=2, dim=1)

        fvl = []
        for i in range(len(fv_norm)):
            fv_hw = fv_norm[i].shape[-2:]
            fl_title = fl_norm.view(*fl_norm.shape[:2], 1, 1).repeat(1, 1, *fv_hw)
            fvl_layer = torch.cat([fv_norm[i], fl_title, generate_coord(fv_norm[i].shape[0], *fv_hw)] if self.coordmap
                                  else [fv_norm[i], fl_title], dim=1)
            fvl.append(fvl_layer)
        return fvl


class ConvBatchNormReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            leaky=False,
            relu=True,
    ):
        super(ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            "bn",
            nn.BatchNorm2d(
                num_features=out_channels, eps=1e-5, momentum=0.999, affine=True
            ),
        )

        if leaky:
            self.add_module("relu", nn.LeakyReLU(0.1))
        elif relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(ConvBatchNormReLU, self).forward(x)


def generate_coord(batch, height, width):
    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    xv, yv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
    xv_min = (xv.float() * 2 - width) / width
    yv_min = (yv.float() * 2 - height) / height
    xv_max = ((xv + 1).float() * 2 - width) / width
    yv_max = ((yv + 1).float() * 2 - height) / height
    xv_ctr = (xv_min + xv_max) / 2
    yv_ctr = (yv_min + yv_max) / 2
    hmap = torch.ones(height, width) * (1. / height)
    wmap = torch.ones(height, width) * (1. / width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0), \
                                               xv_max.unsqueeze(0), yv_max.unsqueeze(0), \
                                               xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0), \
                                               hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch, 1, 1, 1)
    return coord
