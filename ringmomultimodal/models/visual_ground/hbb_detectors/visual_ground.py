from debug_tools import show_value
from ringmomultimodal.models.builder import MULTIMODALDETECTOR
from .single_stage import MultiModalSingleStageHBBDetectorBase
import mmcv
import numpy as np
import torch
from ringmomultimodal.core.visualization.image import imshow_det_bboxes, imshow_gt_det_bboxes


@MULTIMODALDETECTOR.register_module()
class VisualGround(MultiModalSingleStageHBBDetectorBase):
    def show_result(self,
                    img,
                    result,
                    text='',
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    thickness=2,
                    font_size=18,
                    win_name='',
                    show=False,
                    wait_time=0,
                    annotation=None,
                    out_file=None, **kwargs):
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, np.ndarray):
            result = [result]
        bboxes = np.vstack(result)
        if out_file is not None:
            show = False
        # draw bounding boxes
        if annotation is not None:
            img = imshow_gt_det_bboxes(
                img,
                annotation,
                bboxes,
                text=text,
                thickness=thickness,
                font_size=font_size,
                win_name=win_name,
                show=show,
                wait_time=wait_time,
                out_file=out_file)
        else:
            img = imshow_det_bboxes(
                img,
                bboxes,
                bbox_color=bbox_color,
                text_color=text_color,
                text=text,
                thickness=thickness,
                font_size=font_size,
                win_name=win_name,
                show=show,
                wait_time=wait_time,
                out_file=out_file)

        if not (show or out_file):
            return img
