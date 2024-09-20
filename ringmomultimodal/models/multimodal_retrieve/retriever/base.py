from ...builder import build_multimodal, build_similarity_head
from abc import ABCMeta
from mmcv.runner import BaseModule
import torch


class MultimodalRetrieverBase(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 multimodal_backbone,
                 sim_head,
                 init_cfg=None
                 ):
        super(MultimodalRetrieverBase, self).__init__(init_cfg)
        self.multimodal_backbone = build_multimodal(multimodal_backbone)
        self.sim_head = build_similarity_head(sim_head)

    def extract_feat(self, img, text_data, **kwargs):
        """Directly extract features from the backbone+neck."""
        x = self.multimodal_backbone(img, text_data, **kwargs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feats(self, imgs, texts):
        return [self.extract_feat(img, text) for img, text in zip(imgs, texts)]

    def forward_train(self,
                      img_data,
                      text_data,
                      gt_bboxes,
                      img_metas,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs
                      ):
        x = self.extract_feat(text_data=text_data, **img_data, **kwargs)
        losses = self.sim_head.forward_train(x, img_metas, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, imgs, img_metas, text_data, rescale=False, **kwargs):
        feat = self.extract_feat(text_data=text_data, img=imgs, **kwargs)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        results_list = [bbox.detach().cpu().numpy() for bbox in results_list]
        return results_list

    def forward_test(self,
                     imgs,
                     img_metas,
                     text_data,
                     **kwargs
                     ):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')
        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])
        if num_augs == 1:
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            for key in kwargs:
                if isinstance(kwargs[key], list) and len(kwargs[key]) == 1:
                    kwargs[key] = kwargs[key][0]
            return self.simple_test(imgs[0], img_metas[0], text_data, **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            return self.aug_test(imgs, img_metas, **kwargs)

    def forward(self, img_data, text_data, appendix, return_loss=True, ground_truth=None, **kwargs):
        if torch.onnx.is_in_onnx_export():
            assert len(img_data['img_meta']) == 1
            return self.onnx_export(img_data['img'][0], img_data['img_metas'][0])

        if return_loss:
            return self.forward_train(img_data=img_data,
                                      text_data=text_data,
                                      **ground_truth,
                                      **appendix,
                                      **kwargs)
        else:
            return self.forward_test(imgs=img_data['img'],
                                     text_data=text_data,
                                     **appendix,
                                     **kwargs)

    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_data']['img_metas']))

        return outputs

    def val_step(self, data, optimizer=None):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        log_vars_ = dict()
        for loss_name, loss_value in log_vars.items():
            k = loss_name + '_val'
            log_vars_[k] = loss_value

        outputs = dict(
            loss=loss, log_vars=log_vars_, num_samples=len(data['img_data']['img_metas']))

        return outputs
