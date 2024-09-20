# Copyright (c) OpenMMLab. All rights reserved.
from ringmomultimodal.models.builder import MULTIMODALDETECTOR
from .two_stage import MultiModalTwoStageOBBDetectorBase


@MULTIMODALDETECTOR.register_module()
class MultiModalGlidingVertex(MultiModalTwoStageOBBDetectorBase):
    """Implementation of `Gliding Vertex on the Horizontal Bounding Box for
    Multi-Oriented Object Detection <https://arxiv.org/pdf/1911.09358.pdf>`_"""
    pass
