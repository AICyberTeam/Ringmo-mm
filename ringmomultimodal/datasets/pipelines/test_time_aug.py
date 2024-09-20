from mmdet.datasets.pipelines import MultiScaleFlipAug
from ..builder import MULTIMODAL_PIPELINES


@MULTIMODAL_PIPELINES.register_module()
class MultiModalMultiScaleFlipAug(MultiScaleFlipAug):
    def __init__(self,
                 transforms,
                 img_scale=None,
                 scale_factor=None,
                 flip=False,
                 flip_direction='horizontal',
                 aug_test_key=('img_data',),
                 disaug_test_key=('text_data',)
                 ):
        super(MultiModalMultiScaleFlipAug, self).__init__(transforms, img_scale, scale_factor, flip, flip_direction)
        self.disaug_test_key = disaug_test_key

    def __call__(self, results):
        aug_data = []

        flip_args = [(False, None)]
        if self.flip:
            flip_args += [(True, direction)
                          for direction in self.flip_direction]

        for scale in self.img_scale:
            for flip, direction in flip_args:
                _results = results.copy()
                _results[self.scale_key] = scale
                _results['flip'] = flip
                _results['flip_direction'] = direction
                data = self.transforms(_results)
                aug_data.append(data)

        aug_data_dict = {modal_key: {} for modal_key in aug_data[0] if modal_key not in self.disaug_test_key}
        for data in aug_data:
            for modal_key, modal_value in data.items():
                if modal_key in self.disaug_test_key:
                    continue
                for key, val in modal_value.items():
                    if key not in aug_data_dict[modal_key]:
                        aug_data_dict[modal_key][key] = []
                    aug_data_dict[modal_key][key].append(val)
        aug_data_dict.update({modal_key: aug_data[0][modal_key] for modal_key in self.disaug_test_key})
        return aug_data_dict
