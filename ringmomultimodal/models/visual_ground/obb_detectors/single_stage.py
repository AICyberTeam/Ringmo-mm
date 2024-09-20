from mmrotate.models.detectors import RotatedSingleStageDetector
from ringmomultimodal.models.builder import MULTIMODALDETECTOR, build_multimodal


@MULTIMODALDETECTOR.register_module()
class MultiModalSingleStageObbDetectorBase(RotatedSingleStageDetector):
    def __init__(self,
                 multimodal_backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(
            None,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg
        )
        self.multimodal_backbone = build_multimodal(multimodal_backbone)

    def extract_feat(self, input_left, input_right):
        """Directly extract features from the backbone+neck."""
        left_output, right_output = self.multimodal_backbone(input_left, input_right)
        x = [l + r for l, r in zip(left_output, right_output)]
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img, img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses
