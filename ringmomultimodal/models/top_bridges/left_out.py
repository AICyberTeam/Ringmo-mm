from ringmomultimodal.models.bridges.base import RingMoMultiModalBridgeBase
from ringmomultimodal.models.builder import BRIDGE


@BRIDGE.register_module()
class LeftOutTopBridgeBase(RingMoMultiModalBridgeBase):
    def forward(self, left_in_feat, right_in_feat):
        return left_in_feat
