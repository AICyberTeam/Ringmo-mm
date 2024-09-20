from ringmomultimodal.models.builder import MULTIMODALDETECTOR
from .single_stage import MultiModalSingleStageHBBDetectorBase


@MULTIMODALDETECTOR.register_module()
class MultiModalRetinaNet(MultiModalSingleStageHBBDetectorBase):
    pass
