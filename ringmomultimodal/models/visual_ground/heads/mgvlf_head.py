from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

import torch.nn as nn
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, multi_apply)
import torch
from mmcv.runner import force_fp32
from ringmomultimodal.models.builder import BBOX_HEAD
from mmdet.models.builder import build_loss


@BBOX_HEAD.register_module()
class MGVLFHead(BaseDenseHead):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 feat_channels=256,
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 init_cfg=None,
                 **kwargs):
        super(MGVLFHead, self).__init__(init_cfg)
        self.predict_head = nn.Sequential(
            nn.Linear(in_channels, feat_channels),
            nn.ReLU(),
            nn.Linear(feat_channels, 4), )
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

    def init_weights(self):
        for p in self.predict_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feats, img_metas):
        if isinstance(feats, torch.Tensor):
            feats = tuple([feats])
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single, feats, img_metas_list)

    def forward_single(self, x, img_meta):
        outbox = self.predict_head(x)
        outbox = outbox.sigmoid()
        return [outbox]

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas, gt_bboxes_ignore)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
        losses = self.loss(*loss_inputs)
        return losses

    @force_fp32(apply_to=('pred_bbox'))
    def loss(self,
             all_bbox_preds_list,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None, **kwargs):
        num_dec_layers = len(all_bbox_preds_list)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        loss_iou, loss_bbox = multi_apply(
            self.loss_single,
            all_bbox_preds_list,
            all_gt_bboxes_list,
            img_metas_list
        )

        loss_dict = dict()
        loss_dict['loss_iou'] = loss_iou
        loss_dict['loss_bbox'] = loss_bbox
        return loss_dict

    def loss_single(self, bbox_preds, gt_bboxes, img_metas, gt_bbox_ignore=None):
        factors = []
        gt_bboxes = torch.cat(gt_bboxes, 0)
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        bbox_preds = bbox_preds.reshape(-1, 4)
        gt_bboxes_normalized = gt_bboxes / factors
        bbox_cxcy_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bboxes_xyxy_pred = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_xyxy_targets = bbox_cxcywh_to_xyxy(bbox_cxcy_targets) * factors

        loss_iou = self.loss_iou(
            bboxes_xyxy_pred, bboxes_xyxy_targets, None)
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_cxcy_targets, None)
        return loss_iou, loss_bbox

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        pass

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

    def get_bboxes(self,
                   all_bbox_preds_list,
                   img_metas,
                   rescale=False,
                   **kwargs):
        bbox_preds = all_bbox_preds_list
        result_list = []
        for img_id in range(len(img_metas)):
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(self,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        return det_bboxes
