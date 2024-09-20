from abc import ABCMeta
from mmcv.runner import BaseModule
from mmdet.models.builder import MODELS


@MODELS.register_module()
class IdentityBackbone(BaseModule, metaclass=ABCMeta):
    pass


