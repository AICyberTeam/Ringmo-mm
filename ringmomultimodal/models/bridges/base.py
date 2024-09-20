from abc import ABCMeta
from mmcv.runner import BaseModule


class RingMoMultiModalBridgeBase(BaseModule, metaclass=ABCMeta):
    def __init__(self, init_cfg=None):
        super(RingMoMultiModalBridgeBase, self).__init__(init_cfg)

    def forward(self, left_in_feat, right_in_feat, **kwargs):
        pass
