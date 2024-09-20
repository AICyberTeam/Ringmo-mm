# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_detector_by_patches
from .train import train_detector
from .test import *
__all__ = ['inference_detector_by_patches', 'train_detector']
