from .base import RingMoMultiModalBridgeBase
from ringmomultimodal.models.builder import BRIDGE


@BRIDGE.register_module()
class IdentityBridgeBase(RingMoMultiModalBridgeBase):
    def forward(self, left_in_feat, right_in_feat, **kwargs):
        return left_in_feat[-1], right_in_feat
