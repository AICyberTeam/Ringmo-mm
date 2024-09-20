from mmdet.models.detectors import TwoStageDetector
from ringmomultimodal.models.builder import MULTIMODALDETECTOR, build_multimodal


@MULTIMODALDETECTOR.register_module()
class MultiModalTwoStageHBBDetectorBase(TwoStageDetector):
    def __init__(self,
                 multimodal_backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(MultiModalTwoStageHBBDetectorBase, self).__init__(
            backbone=None,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
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
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img, img)
        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses
