# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import tempfile
import time
import warnings
import zipfile
from collections import defaultdict
from functools import partial
import copy
import cv2
import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset
import xml.etree.ElementTree as ET
from mmrotate.core import get_multiscale_patch, merge_results, slide_window
from ringmomultimodal.datasets.builder import MULTIMODAL_DATASETS
from osgeo import gdal
from tqdm import tqdm
import traceback
from ringmomultimodal.core import det_acc
from debug_tools import show_value


@MULTIMODAL_DATASETS.register_module()
class MultiModalRSVGDataset(CustomDataset):
    """DOTA dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """

    PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0), (147, 116, 116), (0, 0, 255), (165, 42, 42),
               (189, 183, 107), (0, 255, 0), (255, 0, 0)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 image_suffix="jpg",
                 filter_small_objs=False,
                 filter_big_objs=False,
                 min_len=300,
                 max_len=500,
                 patch_sizes=[800],
                 patch_steps=[600],
                 img_ratios=[1.0],
                 split_patch=False,
                 negative_labels=['False'],
                 patch_obj_thres=0.5,
                 split_file=None,
                 **kwargs):

        self.patch_sizes = patch_sizes
        self.patch_steps = patch_steps
        self.img_ratios = img_ratios
        self.version = version
        self.difficulty = difficulty
        self.image_suffix = image_suffix.lstrip(".")
        self.filter_small_objs = filter_small_objs
        self.min_len = min_len
        self.filter_big_objs = filter_big_objs
        self.max_len = max_len
        self.split_patch = split_patch
        self.negative_labels = negative_labels
        self.patch_obj_thres = patch_obj_thres
        self.split_file = split_file
        super(MultiModalRSVGDataset, self).__init__(ann_file, pipeline, **kwargs)
        print(f"finished loading {len(self.data_infos)} sample in dataset")

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_xml_file(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        size = root.find("size")
        try:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        except:
            height = None
            width = None
        bboxes = []
        texts = []
        labels = []
        try:
            image_name = root.find("filename").text
        except:
            image_name = os.path.basename(file_path).strip(".xml") + f'.{self.image_suffix}'
        if image_name is None:
            image_name = os.path.basename(file_path).strip(".xml") + f'.{self.image_suffix}'
        # image_name = os.path.join(self.image_suffix, image_name)
        for obj in root.findall('object'):
            bbox = np.array([int(obj[2][0].text), int(obj[2][1].text), int(obj[2][2].text), int(obj[2][3].text)],
                            dtype=np.float32)
            text = obj[3].text
            label = obj[0].text
            bboxes.append(bbox)
            texts.append(text)
            labels.append(label)
        xml_info = dict(
            bboxes=bboxes,
            texts=texts,
            width=width,
            height=height,
            image_name=image_name,
            cls_names=labels
        )
        return xml_info

    def get_gt_in_patch(self, labels, bboxes, patch_coor, thres=0.5):

        x1, y1, x2, y2 = patch_coor
        image_bound = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]], "int32")
        patch_offset = np.array([x1, y1] * 4)
        patch_bboxes = []
        patch_labels = []
        for i, (label, bbox) in enumerate(zip(labels, bboxes)):
            bbox = np.reshape(bbox, [4, 2]).astype("int32")
            if cv2.contourArea(cv2.convexHull(bbox)) < 1.1:
                continue
            # 求两个旋转矩形的交集，返回交点坐标
            intersection = cv2.rotatedRectangleIntersection(cv2.minAreaRect(bbox), cv2.minAreaRect(image_bound))
            if intersection[0] == cv2.INTERSECT_NONE:
                continue
            if cv2.contourArea(cv2.convexHull(intersection[1])) / cv2.contourArea(cv2.convexHull(bbox)) > thres:
                if len(intersection[1]) >= 4:
                    patch_bbox = np.reshape(cv2.boxPoints(cv2.minAreaRect(intersection[1])), [8])
                else:
                    continue
                patch_bboxes.append(patch_bbox - patch_offset)
                patch_labels.append(label)
        return dict(
            bboxes=patch_bboxes,
            cls_names=patch_labels
        )

    def _generate_ann_data_info(self, xml_info, ann_file, img_name, data_infos):
        data_info = {}
        bboxes, cls_names, image_name, texts = \
            xml_info['bboxes'], xml_info['cls_names'], xml_info['image_name'], xml_info['texts']
        data_info['ann'] = {}

        if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
            return None
        data_info['filename'] = img_name

        for bbox, cls_name, text in zip(bboxes, cls_names, texts):
            x_min, y_min, x_max, y_max = list(bbox)
            w = x_max - x_min
            h = y_max - y_min
            if self._filter_len(w, h):
                continue
            gt_bboxes = [bbox]
            gt_labels = [self.cls_map[cls_name]]
            data_info['text'] = text
            if gt_bboxes:
                data_info['ann']['bboxes'] = np.array(
                    gt_bboxes, dtype=np.float32)
                data_info['ann']['labels'] = np.array(
                    gt_labels, dtype=np.int64)
            else:
                data_info['ann']['bboxes'] = np.zeros((0, 4),
                                                      dtype=np.float32)
                data_info['ann']['labels'] = np.array([], dtype=np.int64)

            data_info['ann']['bboxes_ignore'] = np.zeros(
                (0, 4), dtype=np.float32)
            data_info['ann']['labels_ignore'] = np.array(
                [], dtype=np.int64)
            data_info['ann']['polygons_ignore'] = np.zeros(
                (0, 8), dtype=np.float32)

            data_infos.append(copy.deepcopy(data_info))

    def load_annotations(self, ann_folder):
        self.cls_map = {c: i
                        for i, c in enumerate(self.CLASSES)
                        }

        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        assert osp.exists(ann_folder), f'{ann_folder} doesn\'t exist'
        ann_files = glob.glob(ann_folder + '/*.xml')
        with open(self.split_file, 'r') as f:
            ann_in_split = {int(img_idx.strip()) for img_idx in f.readlines()}
        ann_files = list(filter(lambda x: os.path.basename(x).strip('.xml').isdigit()
                                          and int(os.path.basename(x).strip('.xml')) in ann_in_split, ann_files))
        data_infos = []
        if (not ann_files or not os.path.isdir(ann_folder)) and self.test_mode:  # test phase
            if os.path.isdir(ann_folder):
                ann_files = glob.glob(ann_folder + f'/*.{self.image_suffix}')
            else:
                ann_files = [ann_folder]
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.' + self.image_suffix
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []

                if self.split_patch:
                    data_info['gdal_dataset'] = dataset = gdal.Open(osp.join(os.path.dirname(ann_folder), img_name))
                    width = dataset.RasterXSize
                    height = dataset.RasterYSize
                    sizes, steps = get_multiscale_patch(self.patch_sizes, self.patch_steps, self.img_ratios)
                    windows = slide_window(width, height, sizes, steps)
                    for window in windows:
                        data_info_window = data_info.copy()
                        data_info_window['win'] = window
                        data_infos.append(data_info_window)
                else:
                    data_infos.append(data_info)
        else:
            for ann_file in tqdm(ann_files):
                xml_info = self.load_xml_file(ann_file)
                img_id = osp.split(ann_file)[1][:-4]
                bboxes, cls_names, image_name = xml_info['bboxes'], xml_info['cls_names'], xml_info['image_name']
                img_name = img_id + "." + self.image_suffix or image_name
                if self.split_patch:
                    data_info = {}
                    data_info['gdal_dataset'] = dataset = gdal.Open(osp.join(os.path.dirname(ann_folder), img_name))
                    width = dataset.RasterXSize
                    height = dataset.RasterYSize
                    sizes, steps = get_multiscale_patch(self.patch_sizes, self.patch_steps, self.img_ratios)
                    windows = slide_window(width, height, sizes, steps)
                    for window in windows:
                        data_info_window = data_info.copy()
                        data_info_window['win'] = window
                        xml_info_window = xml_info.copy()
                        xml_info_window.update(self.get_gt_in_patch(cls_names, bboxes, window, self.patch_obj_thres))
                        data_info_window.update(self._generate_ann_data_info(xml_info_window, ann_file, img_name))
                        data_infos.append(data_info_window)
                else:
                    self._generate_ann_data_info(xml_info, ann_file, img_name, data_infos)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]

        return data_infos

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        text_info = dict(
            text=self.data_infos[idx]['text'],
            uniq_id=idx
        )
        if self.split_patch:
            results = dict(img_info=img_info, text_info=text_info, win=img_info['win'],
                           gdal_dataset=img_info['gdal_dataset'])
        else:
            results = dict(img_info=img_info, text_info=text_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_train_img(self, idx):
        text_info = dict(
            text=self.data_infos[idx]['text'],
            uniq_id=idx
        )
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        if self.split_patch:
            results = dict(img_info=img_info, ann_info=ann_info, win=img_info['win'], text_info=text_info,
                           gdal_dataset=img_info['gdal_dataset'])
        else:
            results = dict(img_info=img_info, text_info=text_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def _filter_imgs(self, **kwargs):
        """Filter images without ground truths.

        Args:
            **kwargs:
        """
        valid_inds = []

        for i, data_info in enumerate(self.data_infos):
            if 'labels' not in data_info['ann']:
                continue
            if (not self.filter_empty_gt
                    or data_info['ann']['labels'].size > 0):
                valid_inds.append(i)
        return valid_inds

    def _filter_len(self, w, h):
        if self.filter_small_objs and w <= self.min_len and h <= self.min_len:
            return True
        if self.filter_big_objs and w >= self.max_len and h >= self.max_len:
            return True
        return False

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            try:
                data = self.prepare_train_img(idx)
            except Exception as ex:
                print('Data processing exception:', ex)
                traceback.print_exc()
                idx = self._rand_another(idx)
                continue
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def get_text(self, idx):
        return self.data_infos[idx]['text']

    def evaluate(self,
                 results,
                 metric='DetAcc',
                 logger=None,
                 level_threshold=np.linspace(0.5, 0.9, 5),
                 ):
        if "detacc" in metric.lower():
            gt_bboxes = [self.get_ann_info(i)['bboxes'] for i in range(len(self))]
            # show_value(results)
            results = np.array(results)
            acc = det_acc(results, gt_bboxes, logger=logger, metric=metric)
        else:
            raise NotImplementedError
        return {metric: acc}


if __name__ == '__main__':
    from mmdet.datasets.pipelines.loading import LoadImageFromFile
    from ringmomultimodal.datasets.pipelines import VoidMask, BertTextTokenize
    from ringmomultimodal.datasets.pipelines.formatting import MultiModalCollect
    from mmdet.datasets.pipelines import LoadAnnotations, Resize, RandomFlip, Normalize, Pad, DefaultFormatBundle

    pipeline = [
        LoadImageFromFile(),
        LoadAnnotations(with_bbox=True),
        BertTextTokenize(
            bert_model_path='/mnt/sdb/share1416/airalgorithm/code/ringmoMultiModal/ringmouscframework/saved_models'),
        Resize(img_scale=(512, 512), keep_ratio=True),
        RandomFlip(flip_ratio=0.5),
        Normalize(
            **dict(
                mean=[122.52188731, 122.52188731, 122.52188731],
                std=[63.545590175277034, 63.545590175277034, 63.545590175277034],
                to_rgb=True
            )
        ),
        Pad(size_divisor=32),
        VoidMask(),
        DefaultFormatBundle(),
        MultiModalCollect(
            text_keys=('word_id', 'word_mask'),
            gt_keys=('gt_bboxes', 'gt_labels'),
            appendix_keys=('word_mask', 'parse_out', 'img_metas'),
            img_data_keys=('img', 'gt_bboxes', 'gt_labels'),
            img_meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg'))
    ]
    from torchvision.transforms.transforms import Compose

    class_names = ['vehicle', 'overpass', 'chimney', 'Expressway-Service-area', 'harbor', 'basketballcourt',
                   'baseballfield', 'groundtrackfield', 'bridge', 'stadium', 'windmill', 'airplane', 'tenniscourt',
                   'airport', 'ship', 'trainstation', 'storagetank', 'golffield', 'dam', 'Expressway-toll-station']
    dataset = MultiModalRSVGDataset(
        # ann_file='/mnt/sdb/share1416/airalgorithm/datasets/DIOR_RSVG/Annotations/',
        ann_file="/mnt/sdb/share1416/Ringmo_VG/MGVLF/RSVG-pytorch-main/RSVG-pytorch-main/DIOR_RSVG/Annotations",
        pipeline=pipeline,
        # img_prefix="/mnt/sdb/share1416/airalgorithm/datasets/DIOR_RSVG/"
        img_prefix="/mnt/sdb/share1416/Ringmo_VG/MGVLF/RSVG-pytorch-main/RSVG-pytorch-main/DIOR_RSVG/JPEGImages/",
        classes=class_names,
        split_file='/mnt/sdb/share1416/Ringmo_VG/MGVLF/RSVG-pytorch-main/RSVG-pytorch-main/DIOR_RSVG/val.txt'
    )


    def show_dict(dic, level=0):
        for key, value in dic.items():
            if isinstance(value, dict) and len(value.keys()) > 0:
                print(f"{'     ' * level}{key}:")
                show_dict(value, level + 1)
            else:
                print(f"{'     ' * level}{key}: {value}")


    for data in dataset:
        show_dict(data)
