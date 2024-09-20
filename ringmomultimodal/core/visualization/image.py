# Copyright (c) OpenMMLab. All rights reserved.
import sys

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as mask_util
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from debug_tools import show_value
from mmdet.core.evaluation.panoptic_utils import INSTANCE_OFFSET
from mmdet.core.mask.structures import bitmap_to_polygon
from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization.palette import get_palette, palette_val
from mmdet.core.visualization.image import draw_bboxes, imshow_gt_det_bboxes, _get_adaptive_scales


def draw_title_texts(ax,
                     text='',
                     color='w',
                     font_size=20,
                     horizontal_alignment='left'):
    font_size_mask = font_size
    ax.text(
        0, 0,
        f'{text}',
        bbox={
            'facecolor': 'black',
            'alpha': 0.8,
            'pad': 0.7,
            'edgecolor': 'none'
        },
        color=color,
        fontsize=font_size_mask,
        verticalalignment='top',
        horizontalalignment=horizontal_alignment)

    return ax


def draw_bbox_label(ax,
                    texts,
                    positions,
                    scores=None,
                    color='w',
                    font_size=20,
                    scales=None,
                    horizontal_alignment='left'):
    for i, (pos, text) in enumerate(zip(positions, texts)):
        label_text = text
        if scores is not None:
            label_text += f'|{scores[i]:.02f}'
        text_color = color[i] if isinstance(color, list) else color

        font_size_mask = font_size if scales is None else font_size * scales[i]
        ax.text(
            pos[0],
            pos[1],
            f'{text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=font_size_mask,
            verticalalignment='top',
            horizontalalignment=horizontal_alignment)

    return ax


EPS = 1e-2


def imshow_det_bboxes(img,
                      bboxes=None,
                      text='',
                      label='predict',
                      bbox_color='green',
                      text_color='black',
                      thickness=2,
                      font_size=8,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):
    assert bboxes is None or bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert bboxes is None or (bboxes.shape[1] == 4 or bboxes.shape[1] == 5), \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'

    img = mmcv.imread(img).astype(np.uint8)

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)

    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    text_palette = palette_val(get_palette(text_color, 1))
    text_colors = text_palette[0]

    num_bboxes = 0
    if bboxes is not None:
        num_bboxes = bboxes.shape[0]
        bbox_color = palette_val(get_palette(bbox_color, 1))[0]
        colors = [bbox_color for _ in bboxes[:num_bboxes]]
        draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)

        horizontal_alignment = 'left'
        draw_title_texts(
            ax,
            text,
            color=text_colors,
            font_size=font_size,
            horizontal_alignment=horizontal_alignment)
        positions = bboxes[:, :2].astype(np.int32) + thickness
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)
        scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None

        draw_bbox_label(ax,
                        [label],
                        positions,
                        scores=scores,
                        color=text_colors,
                        font_size=8,
                        scales=scales,
                        horizontal_alignment=horizontal_alignment)
    plt.title(win_name)
    plt.imshow(img)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    if sys.platform == 'darwin':
        width, height = canvas.get_width_height(physical=True)
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img


def imshow_gt_det_bboxes(img,
                         annotation,
                         result,
                         text='',
                         gt_bbox_color=(61, 102, 255),
                         det_bbox_color=(241, 101, 72),
                         thickness=2,
                         font_size=25,
                         win_name='',
                         show=True,
                         wait_time=0,
                         out_file=None,
                         overlay_gt_pred=True):
    gt_bboxes = annotation['bboxes']
    img = mmcv.imread(img)
    img_with_gt = imshow_det_bboxes(
        img,
        gt_bboxes,
        label='ground_truth',
        text=text,
        bbox_color=gt_bbox_color,
        text_color=gt_bbox_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=False)
    bboxes = np.vstack(result)
    if overlay_gt_pred:
        img = imshow_det_bboxes(
            img_with_gt,
            bboxes,
            label='predict',
            bbox_color=det_bbox_color,
            text_color=det_bbox_color,
            text=text,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=int(wait_time),
            out_file=out_file)
    else:
        img_with_det = imshow_det_bboxes(
            img,
            bboxes,
            bbox_color=det_bbox_color,
            text_color=det_bbox_color,
            text=text,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=False)
        img = np.concatenate([img_with_gt, img_with_det], axis=0)

        plt.imshow(img)
        if show:
            if wait_time == 0:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)
        plt.close()

    return img
