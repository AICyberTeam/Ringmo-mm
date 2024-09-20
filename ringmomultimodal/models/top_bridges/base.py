from torch import nn
from mmcv.runner import BaseModule


class TopBridgeBase(BaseModule):
    def __init__(self, init_cfg=None):
        super(TopBridgeBase, self).__init__(init_cfg)

    def forward(self, fv, fl, **kwargs):
        pass
