import numpy as np
from terminaltables import AsciiTable
from mmcv.utils import print_log

def element_wise_iou(bboxes1, bboxes2):
    '''
    predict_bboxes \in (n, 4),
    gt_bboxes \in (n, 4)
    '''
    x_min = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
    y_min = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
    x_max = np.minimum(bboxes1[:, 2], bboxes2[:, 2])
    y_max = np.minimum(bboxes1[:, 3], bboxes2[:, 3])
    inter_w = np.maximum(x_max - x_min, 0)
    inter_h = np.maximum(y_max - y_min, 0)
    inter_area = inter_w * inter_h
    union_area =(bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1]) - inter_area
    return inter_area / union_area

def det_acc(predict_bboxes, gt_bboxes, level_thresholds=np.linspace(0.5, 0.9, 5), logger=None, metric="detAcc@0.5"):
    assert len(predict_bboxes) == len(gt_bboxes), "The number of predicted bboxes and ground-truths' should be same."
    if isinstance(predict_bboxes, list):
        predict_bboxes = np.concatenate(predict_bboxes, axis=0)
    if isinstance(gt_bboxes, list):
        gt_bboxes = np.concatenate(gt_bboxes, axis=0)
    ious = element_wise_iou(predict_bboxes, gt_bboxes)
    total_num = ious.shape[0]
    acc = dict()
    for thres in level_thresholds:
        precision = np.sum(ious > thres)
        acc[thres] = dict(precision=precision, total=total_num, precise_rate=precision/total_num)
    standard = float(metric.split("@")[-1])
    print_det_acc_summary(acc, logger, standard)
    return acc[standard]['precise_rate']

def print_det_acc_summary(acc_info, logger, standard):
    header = ['threshold', 'precision', 'total', 'precision rate']
    table_data = [header]
    for threshold, info in acc_info.items():
        row_data = [f'acc@{threshold}', f"{info['precision']}", f"{info['total']}", f"{info['precise_rate']:.3f}"]
        table_data.append(row_data)
    info = acc_info[standard]
    last_row = [f'acc@{standard}', f"{info['precision']}", f"{info['total']}", f"{info['precise_rate']:.3f}"]
    table_data.append(last_row)
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)


