from .base import RingMoMultiModalBase
from ringmomultimodal.models.builder import MULTIMODAL


@MULTIMODAL.register_module()
class SimpleMultiModal(RingMoMultiModalBase):
    pass
