from ..builder import MULTIMODAL_PIPELINES
from mmcv.parallel import DataContainer as DC
from debug_tools import _show_dict


@MULTIMODAL_PIPELINES.register_module()
class MultiModalCollect:
    def __init__(self,
                 text_keys=('word_id', 'word_mask'),
                 gt_keys=(),
                 appendix_keys=('word_mask', 'parse_out', 'img_metas'),
                 img_data_keys=('img', 'gt_bboxes', 'gt_labels'),
                 img_meta_keys=('filename', 'ori_filename', 'ori_shape',
                                'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                'flip_direction', 'img_norm_cfg')):
        self.img_meta_keys = img_meta_keys
        self.img_data_keys = img_data_keys
        self.text_keys = text_keys
        self.gt_keys = gt_keys
        self.appendix_keys = appendix_keys

    def __call__(self, results):
        img_meta = {}
        for key in self.img_meta_keys:
            img_meta[key] = results[key]
        results['img_metas'] = DC(img_meta, cpu_only=True)

        final_results = dict(img_data=dict(key_names=self.img_data_keys),
                             text_data=dict(key_names=self.text_keys),
                             ground_truth=dict(key_names=self.gt_keys),
                             appendix=dict(key_names=self.appendix_keys))

        for key, value in final_results.items():
            if 'key_names' not in value:
                continue
            key_names = value.pop('key_names')
            if not key_names:
                continue
            for key_name in key_names:
                value[key_name] = results[key_name]
            final_results[key] = value
        return final_results
