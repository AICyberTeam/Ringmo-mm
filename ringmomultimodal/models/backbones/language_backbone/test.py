from ....models.builder import BACKBONE
from ..utils import FlowBackBoneBase
import torch


@BACKBONE.register_module()
class Test(FlowBackBoneBase):
    def __init__(self, dim=768):
        super(Test, self).__init__(None)
        self.gap = torch.nn.AdaptiveAvgPool1d(dim)
        self.emb = torch.nn.Parameter(torch.randn(4, 40, 768))

    def forward(self, word_id, word_mask):
        return self.emb
