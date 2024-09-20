﻿# Copyright (c) OpenMMLab. All rights reserved.
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
import cv2
import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset
import xml.etree.ElementTree as ET
from mmrotate.core import get_multiscale_patch, merge_results, slide_window
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from .builder import ROTATED_DATASETS
from osgeo import gdal
from tqdm import tqdm
def load_xml_file(file_path):
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
    cls_names = []
    try:
        image_name = root.find("source/filename")
    except:
        image_name = None
    for obj in root.findall('objects/object'):
        bbox = np.array(
            list(map(
                lambda x: list(map(float, x.text.split(','))),
                obj.findall('points/point')
            )),
            dtype=np.float32
        ).reshape(-1)
        bboxes.append(bbox)
        cls_name = obj.find('possibleresult/name').text
        cls_names.append(cls_name)
    xml_info = dict(
        bboxes=bboxes,
        cls_names=cls_names,
        width=width,
        height=height,
        image_name=image_name
    )
    return xml_info

def get_gt_in_patch(labels, bboxes, patch_coor, thres=0.5):

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

@ROTATED_DATASETS.register_module()
class AircasDataset(CustomDataset):
    """DOTA dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """
    CLASSES=[]
    # CLASSES = ['Truck Tractor', 'Warship', 'Small Car', 'Dump Truck', 'A220', 'Basketball Court',
    # 'Tugboat', 'Bus', 'ARJ21', 'Passenger Ship', 'A321', 'Cargo Truck', 'Roundabout', 'Fishing Boat',
    # 'other-airplane', 'Liquid Cargo Ship', 'Engineering Ship', 'Baseball Field', 'Boeing777', 'Van', 'A330',
    # 'Motorboat', 'Trailer', 'Tractor', 'Intersection', 'Tennis Court', 'Excavator', 'other-ship', 'Dry Cargo Ship',
    # 'Football Field', 'other-vehicle', 'Boeing747', 'Bridge', 'Boeing737', 'Boeing787', 'A350', 'C919']

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
                 image_suffix="png",
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
                 **kwargs):

        self.patch_sizes=patch_sizes
        self.patch_steps=patch_steps
        self.img_ratios=img_ratios
        self.version = version
        self.difficulty = difficulty
        self.image_suffix = image_suffix.lstrip(".")
        self.filter_small_objs = filter_small_objs
        self.min_len=min_len
        self.filter_big_objs=filter_big_objs
        self.max_len=max_len
        self.split_patch = split_patch
        self.negative_labels = negative_labels
        self.patch_obj_thres = patch_obj_thres
        super(AircasDataset, self).__init__(ann_file, pipeline, **kwargs)
        print(f"finished loading {len(self.data_infos)} sample in dataset")


    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def _generate_ann_data_info(self, xml_info, ann_file, img_name):
        data_info = {}
        bboxes, cls_names, image_name = xml_info['bboxes'], xml_info['cls_names'], xml_info['image_name']
        data_info['ann'] = {}
        gt_bboxes = []
        gt_labels = []
        gt_polygons = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
        gt_polygons_ignore = []

        if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
            return None
        data_info['filename'] = img_name

        for bbox_info, cls_name in zip(bboxes, cls_names):
            poly = np.array(bbox_info[:8], dtype=np.float32)
            try:
                x, y, w, h, a = poly2obb_np(poly, self.version)
            except:  # noqa: E722
                continue
            label = self.cls_map[cls_name]

            ##如果确定要过滤小目标，并且该框没有被过滤掉，就加入到gt_bboxes中
            if self._filter_len(w, h):
                continue
            gt_bboxes.append([x, y, w, h, a])
            gt_labels.append(label)
            gt_polygons.append(poly)

        if gt_bboxes:
            data_info['ann']['bboxes'] = np.array(
                gt_bboxes, dtype=np.float32)
            data_info['ann']['labels'] = np.array(
                gt_labels, dtype=np.int64)
            data_info['ann']['polygons'] = np.array(
                gt_polygons, dtype=np.float32)
        # 如果gt_bboxes为空，各个属性都写成0
        else:
            data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                  dtype=np.float32)
            data_info['ann']['labels'] = np.array([], dtype=np.int64)
            data_info['ann']['polygons'] = np.zeros((0, 8),
                                                    dtype=np.float32)

        if gt_polygons_ignore:
            data_info['ann']['bboxes_ignore'] = np.array(
                gt_bboxes_ignore, dtype=np.float32)
            data_info['ann']['labels_ignore'] = np.array(
                gt_labels_ignore, dtype=np.int64)
            data_info['ann']['polygons_ignore'] = np.array(
                gt_polygons_ignore, dtype=np.float32)
        else:
            data_info['ann']['bboxes_ignore'] = np.zeros(
                (0, 5), dtype=np.float32)
            data_info['ann']['labels_ignore'] = np.array(
                [], dtype=np.int64)
            data_info['ann']['polygons_ignore'] = np.zeros(
                (0, 8), dtype=np.float32)
        return data_info


    def load_annotations(self, ann_folder):

        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        self.cls_map = {c: i
                        for i, c in enumerate(self.CLASSES)
                        }  # in mmdet v2.0 label is 0-based
        assert osp.exists(ann_folder) , f'{ann_folder} doesn\'t exist'
        ann_files = glob.glob(ann_folder + '/*.xml')
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
                xml_info = load_xml_file(ann_file)
                img_id = osp.split(ann_file)[1][:-4]
                bboxes, cls_names, image_name = xml_info['bboxes'], xml_info['cls_names'], xml_info['image_name']
                img_name =  img_id + "." + self.image_suffix or image_name
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
                        xml_info_window.update(get_gt_in_patch(cls_names, bboxes, window, self.patch_obj_thres))
                        data_info_window.update(self._generate_ann_data_info(xml_info_window, ann_file, img_name))
                        data_infos.append(data_info_window)
                else:
                    data_info = self._generate_ann_data_info(xml_info, ann_file, img_name)
                    data_infos.append(data_info)


        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]

        return data_infos

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        if self.split_patch:
            results = dict(img_info=img_info, win=img_info['win'], gdal_dataset=img_info['gdal_dataset'])
        else:
            results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        if self.split_patch:
            results = dict(img_info=img_info, ann_info=ann_info, win=img_info['win'], gdal_dataset=img_info['gdal_dataset'])
        else:
            results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        #print(annotations)
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError

        return eval_results

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

    def _filter_len(self,w,h):
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


    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Args:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.

        Returns:
            list: merged results.
        """

        def extract_xy(img_id):
            """Extract x and y coordinates from image ID.

            Args:
                img_id (str): ID of the image.

            Returns:
                Tuple of two integers, the x and y coordinates.
            """
            pattern = re.compile(r'__(\d+)___(\d+)')
            match = pattern.search(img_id)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                return x, y
            else:
                warnings.warn(
                    "Can't find coordinates in filename, "
                    'the coordinates will be set to (0,0) by default.',
                    category=Warning)
                return 0, 0

        collector = defaultdict(list)
        for idx, img_id in enumerate(self.img_ids):
            result = results[idx]
            oriname = img_id.split('__', maxsplit=1)[0]
            x, y = extract_xy(img_id)
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))
            new_result = np.concatenate(new_result, axis=0)
            collector[oriname].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Executing on Single Processor')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print(f'Executing on {nproc} processors')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        # Return a zipped list of merged results
        return zip(*merged_results)

    def _results2submission(self, id_list, dets_list, out_folder=None):
        """Generate the submission of full images.

        Args:
            id_list (list): Id of images.
            dets_list (list): Detection results of per class.
            out_folder (str, optional): Folder of submission.
        """
        if osp.exists(out_folder):
            raise ValueError(f'The out_folder should be a non-exist path, '
                             f'but {out_folder} is existing')
        os.makedirs(out_folder)

        files = [
            osp.join(out_folder, 'Task1_' + cls + '.txt')
            for cls in self.CLASSES
        ]
        file_objs = [open(f, 'w') for f in files]
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for f, dets in zip(file_objs, dets_per_cls):
                if dets.size == 0:
                    continue
                bboxes = obb2poly_np(dets, self.version)
                for bbox in bboxes:
                    txt_element = [img_id, str(bbox[-1])
                                   ] + [f'{p:.2f}' for p in bbox[:-1]]
                    f.writelines(' '.join(txt_element) + '\n')

        for f in file_objs:
            f.close()

        target_name = osp.split(out_folder)[-1]
        with zipfile.ZipFile(
                osp.join(out_folder, target_name + '.zip'), 'w',
                zipfile.ZIP_DEFLATED) as t:
            for f in files:
                t.write(f, osp.split(f)[-1])

        return files

    def format_results(self, results, submission_dir=None, nproc=4, **kwargs):
        """Format the results to submission text (standard format for DOTA
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            submission_dir (str, optional): The folder that contains submission
                files. If not specified, a temp folder will be created.
                Default: None.
            nproc (int, optional): number of process.

        Returns:
            tuple:

                - result_files (dict): a dict containing the json filepaths
                - tmp_dir (str): the temporal directory created for saving \
                    json files when submission_dir is not specified.
        """
        nproc = min(nproc, os.cpu_count())
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            f'The length of results is not equal to '
            f'the dataset len: {len(results)} != {len(self)}')
        if submission_dir is None:
            submission_dir = tempfile.TemporaryDirectory()
        else:
            tmp_dir = None

        print('\nMerging patch bboxes into full image!!!')
        start_time = time.time()
        id_list, dets_list = self.merge_det(results, nproc)
        stop_time = time.time()
        print(f'Used time: {(stop_time - start_time):.1f} s')

        result_files = self._results2submission(id_list, dets_list,
                                                submission_dir)

        return result_files, tmp_dir


def _merge_func(info, CLASSES, iou_thr):
    """Merging patch bboxes into full image.

    Args:
        CLASSES (list): Label category.
        iou_thr (float): Threshold of IoU.
    """
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)

    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    big_img_results = []
    for i in range(len(CLASSES)):
        if len(dets[labels == i]) == 0:
            big_img_results.append(dets[labels == i])
        else:
            try:
                cls_dets = torch.from_numpy(dets[labels == i]).cuda()
            except:  # noqa: E722
                cls_dets = torch.from_numpy(dets[labels == i])
            nms_dets, keep_inds = nms_rotated(cls_dets[:, :5], cls_dets[:, -1],
                                              iou_thr)
            big_img_results.append(nms_dets.cpu().numpy())
    return img_id, big_img_results
