from mmcv.runner import BaseModule
import inspect


class FlowBackBoneBase(BaseModule):
    def __init__(self, init_cfg=None):
        super(FlowBackBoneBase, self).__init__(init_cfg=init_cfg)
        self.components = [self.forward]
