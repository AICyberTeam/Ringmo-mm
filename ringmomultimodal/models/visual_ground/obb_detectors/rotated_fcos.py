# Copyright (c) OpenMMLab. All rights reserved.
from ringmomultimodal.models.builder import MULTIMODALDETECTOR
from .single_stage import MultiModalSingleStageObbDetectorBase


@MULTIMODALDETECTOR.register_module()
class MultiModalRotatedFCOS(MultiModalSingleStageObbDetectorBase):
    """Implementation of Rotated `FCOS.`__

    __ https://arxiv.org/abs/1904.01355
    """
    pass
