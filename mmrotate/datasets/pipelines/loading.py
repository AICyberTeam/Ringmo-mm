# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmdet.datasets.pipelines import LoadImageFromFile

from ..builder import ROTATED_PIPELINES
import time

@ROTATED_PIPELINES.register_module()
class LoadPatchFromImage(LoadImageFromFile):
    """Load an patch from the huge image.

    Similar with :obj:`LoadImageFromFile`, but only reserve a patch of
    ``results['img']`` according to ``results['win']``.
    """

    def __call__(self, results, int_16_norm=True):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with image in ``results['img']``.

        Returns:
            dict: The dict contains the loaded patch and meta information.
        """

        x_start, y_start, x_stop, y_stop = results['win']
        width = x_stop - x_start
        height = y_stop - y_start
        if "gdal_dataset" in results and "gdal_dataset" is not None:
           patch = dataset.ReadAsArray(x_start, y_start, width, height).transpose(1, 2, 0)
        else:
            img = results['img']
            patch = img[y_start:y_stop, x_start:x_stop].copy()
        if patch.dtype == 'uint16' and int_16_norm:
            s_t = time.time()
            min_val, max_val = patch.min(), patch.max()
            #print('max_min: ', time.time() - s_t)
            patch = ((patch - min_val) / (max_val - min_val) * 255)#.astype('uint8')

            #print('norm: ', time.time() - s_t)
            #print("Effect")
        if height > patch.shape[0] or width > patch.shape[1]:
            patch = mmcv.impad(patch, shape=(height, width))

        if self.to_float32:
            patch = patch.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = patch
        results['img_shape'] = patch.shape
        results['ori_shape'] = patch.shape
        results['img_fields'] = ['img']
        return results
