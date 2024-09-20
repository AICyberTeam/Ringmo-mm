# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
import time
from mmrotate.core import get_multiscale_patch, merge_results, slide_window
from osgeo import gdal
import cv2
def inference_detector_by_patches(model,
                                  img,
                                  sizes,
                                  steps,
                                  ratios,
                                  merge_iou_thr,
                                  bs=1,
                                  to_gray=False
                                  ):
    """inference patches with the detector.

    Split huge image(s) into patches and inference them with the detector.
    Finally, merge patch results on one huge image by nms.

    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray or): Either an image file or loaded image.
        sizes (list): The sizes of patches.
        steps (list): The steps between two patches.
        ratios (list): Image resizing ratios for multi-scale detecting.
        merge_iou_thr (float): IoU threshold for merging results.
        bs (int): Batch size, must greater than or equal to 1.

    Returns:
        list[np.ndarray]: Detection results.
    """
    assert bs >= 1, 'The batch size must greater than or equal to 1'
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    cfg = cfg.copy()
    # set loading pipeline type
    cfg.data.test.pipeline[0].type = 'LoadPatchFromImage'
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    if  not img.endswith('tif'):
        img = mmcv.imread(img)
        width, height = img.shape[:2]
    else:
        dataset = gdal.Open(img)
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        img = dataset.ReadAsArray(0, 0, width, height)
        if to_gray and len(img.shape) > 2 and img.shape[0] > 1:
            img = cv2.cvtColor(img.transpose(1,2,0), cv2.COLOR_BGR2GRAY) 
        if len(img.shape) != 3:
            img = img.reshape(1, *img.shape)
        if img.shape[0] == 1:
            img = img.repeat(3, 0)
        img = img.transpose(1, 2, 0)  # c, h, w

    #print(img)
    sizes, steps = get_multiscale_patch(sizes, steps, ratios)
    windows = slide_window(width, height, sizes, steps)

    results = []
    start = 0
    start_time = time.time()
    while True:
        # prepare patch data
        patch_datas = []
        if (start + bs) > len(windows):
            end = len(windows)
        else:
            end = start + bs
        for window in windows[start:end]:
            data = dict(img=img, win=window.tolist())
            #print(data)
            #raise EOFError
            data = test_pipeline(data)
            #print('pass')
            patch_datas.append(data)
        data = collate(patch_datas, samples_per_gpu=len(patch_datas))
        # just get the actual data from DataContainer
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
        data['img'] = [img.data[0] for img in data['img']]
        #print(data.keys())
        #raise EOFError
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            for m in model.modules():
                assert not isinstance(
                    m, RoIPool
                ), 'CPU inference with RoIPool is not supported currently.'

        # forward the model
        with torch.no_grad():
            results.extend(model(return_loss=False, rescale=True, **data))

        end_time = time.time()
        during_time = end_time - start_time
        qps = len(results)/during_time
        print('qps:', qps)
        if end >= len(windows):
            break
        start += bs

    results = merge_results(
        results,
        windows[:, :2],
        img_shape=(width, height),
        iou_thr=merge_iou_thr,
        device=device)
    return results
