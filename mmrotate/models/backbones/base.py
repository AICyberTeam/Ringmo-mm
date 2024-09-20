# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy
import random
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout

from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.utils import to_2tuple
from ..builder import ROTATED_BACKBONES as BACKBONES
from ...utils import get_root_logger


class BackboneBase(BaseModule):

    def __init__(self):
        super(BackboneBase, self).__init__()
        self.stages = ModuleList()










