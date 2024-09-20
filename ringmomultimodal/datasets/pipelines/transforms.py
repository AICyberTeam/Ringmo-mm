from ..builder import MULTIMODAL_PIPELINES
import torch


@MULTIMODAL_PIPELINES.register_module()
class VoidMask(object):

    def __init__(self):
        pass

    def __call__(self, results):
        results["img_mask"] = torch.zeros(*results['img'].shape[:2])  # .permute(3,1,2)
        return results
