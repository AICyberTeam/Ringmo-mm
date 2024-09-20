# -*- coding: utf-8 -*-
"""
obb标注裁剪流程
1.obb保证为矩形：non_rectangle_to_rectangle，使用cv2.minAreaRect求最小外接矩，使用cv2.boxPoints得到最小外接矩的4个角点
2.设置步长和窗口大小，进行滑窗裁剪
3.如果该patch中不含有目标，则放弃该patch，不对该patch进行训练；提取被滑窗截断的有效obb框
4.将在大图中的obb框坐标转换为patch中的obb坐标
5.文件落盘，保存patch和xml；如需可视化则再进行可视化检查
"""
import os
import cv2
import xml
import argparse
import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image
from func_utils import *
from osgeo import gdal, gdalconst
from xml.dom.minidom import Document


def read_aircas_xml(xml_path):
    """
    读取AIRCAS格式中xml的信息
    :param xml_path: 
    :return: 
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    try:
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        image_name = root.find("source/filename").text
    except:
        width = None
        height = None
        image_name = None

    bboxes = []
    cls_names = []
    for obj in root.findall("objects/object"):
        # 读取obb坐标
        bbox = np.array(list(map(
            lambda x: list(map(float, getattr(x, "text").split(","))),
            obj.findall("points/point"))),
            dtype=np.float32
        ).reshape(-1)
        # # AIRCAS格式第一个点和最后一个点是相同的
        # assert set(bbox[0:2]) == set(bbox[-2:]), "bbox is not closed"
        # 前8个点即该obb的4个角点坐标
        bbox = bbox[0:8]
        bboxes.append(bbox)
        # 读取类别名
        cls_name = obj.find("possibleresult/name").text
        cls_names.append(cls_name)

    # 整合xml信息
    xml_info = dict(
        bboxes=bboxes,
        cls_names=cls_names,
        width=width,
        height=height,
        image_name=image_name
    )

    return xml_info


def dist(p1, p2):
    """
    计算平面中两个点之间的欧式距离
    :param p1: 
    :param p2: 
    :return: 
    """
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def select_non_rectangle(raw_obb):
    """
    如果是矩形，则中心点到4个角点的距离相等
    但标注有时会存在取舍误差，中心点到4个角点的距离可能会存在微小误差，这也都会算到非矩形标注中，不过不影响训练
    :param raw_obb: [[x00, y00, x01, y01, x02, y02, x03, y03], [x10, y10, x11, y11, x12, y12, x13, y13], ...]
    :return: 非矩形obb的索引
    """
    non_rectangle_idx = []
    for i in range(len(raw_obb)):
        # 如果标注点不足或较多，则直接append该索引
        if len(raw_obb[i]) != 8:
            non_rectangle_idx.append(i)
            continue

        center_x = (raw_obb[i][0] + raw_obb[i][2] + raw_obb[i][4] + raw_obb[i][6]) / 4
        center_y = (raw_obb[i][1] + raw_obb[i][3] + raw_obb[i][5] + raw_obb[i][7]) / 4
        d1 = dist((raw_obb[i][0], raw_obb[i][1]), (center_x, center_y))
        d2 = dist((raw_obb[i][2], raw_obb[i][3]), (center_x, center_y))
        d3 = dist((raw_obb[i][4], raw_obb[i][5]), (center_x, center_y))
        d4 = dist((raw_obb[i][6], raw_obb[i][7]), (center_x, center_y))

        if not (d1 == d2 and d1 == d3 and d1 == d4):
            non_rectangle_idx.append(i)
            # print("第{}个框为非矩形标注".format(i))
    return non_rectangle_idx


def non_rectangle_to_rectangle(obb_points, non_rectangle_idx):
    """
    将非矩形标注转换为矩形标注
    但原本就是矩形标注则无需进行转换
    :param obb_points: [[x00, y00, x01, y01, x02, y02, x03, y03], [x10, y10, x11, y11, x12, y12, x13, y13], ...]
    :param non_rectangle_idx: 非矩形框的索引
    :return: 转换为矩形的obb坐标，非矩形标注转换为矩形标注后数据类型会变为np.float32，所以需要将其会转换为np.int32
    """
    for idx in non_rectangle_idx:
        tmp = np.array(obb_points[idx], dtype=np.int32)
        tmp = tmp.reshape([-1, 2])
        # cv2.minAreaRect求一组点的最小外接矩
        rect1 = cv2.minAreaRect(tmp)
        x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
        # cv2.boxPoints返回矩形4个角点，从y最高的点开始按顺时针顺序排列
        box = cv2.boxPoints(((x, y), (w, h), theta))
        box = np.reshape(box, [-1, ]).astype(np.int32)
        obb_points[idx] = [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]

    return obb_points


def is_bboxes_in_patch(patch_coor, obj_range):
    """
    该切片中是否包含目标框
    :param patch_coor: 切片在大图中的位置
    :param obj_range: 原始大图中所有目标的范围
    :return: 
    """
    ixmin = np.maximum(patch_coor[0], obj_range[0])
    iymin = np.maximum(patch_coor[1], obj_range[1])
    ixmax = np.minimum(patch_coor[2], obj_range[2])
    iymax = np.minimum(patch_coor[3], obj_range[3])
    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)

    inters = iw * ih      # 所有obb框与该切片的交集面积

    return inters


def select_bbox(_bbox, image_shape, iou_threshold=0.5):
    """
    cv2.convexHull()求最小凸包，返回最小凸包的顶点的坐标
    cv2.contourArea()计算给定轮廓的面积
    cv2.rotatedRectangleIntersection()求两个旋转矩形的交集，返回交点坐标
    cv2.minAreaRect()求一组点的最小外接矩
    :param _bbox: 大图中所有obb框，[[x00, y00, x01, y01, x02, y02, x03, y03], [x10, y10, x11, y11, x12, y12, x13, y13], ...]
    :param image_shape: 当前切片坐标
    :return: 
    """
    bbox = np.copy(_bbox)
    x1, y1, x2, y2 = image_shape
    image_bound = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]], "int32")
    new_bb = []
    new_idx = []

    for idx, bb in enumerate(bbox):
        bb = np.reshape(bb, [4, 2]).astype("int32")
        # 去除标注失误。如果标注时只标了一个点，没有形成矩形，则轮廓面积为1，所以应该去除此种标注
        if cv2.contourArea(cv2.convexHull(bb)) < 1.1:
            continue

        # 求两个旋转矩形的交集，返回交点坐标
        intersection = cv2.rotatedRectangleIntersection(cv2.minAreaRect(bb), cv2.minAreaRect(image_bound))
        # 如果该切片和当前obb框无交集，则继续遍历下一个obb框
        if intersection[0] == cv2.INTERSECT_NONE:
            continue
        # 如果交并比多于iou_threshold，通常iou_threshold为0.5
        if cv2.contourArea(cv2.convexHull(intersection[1])) / cv2.contourArea(cv2.convexHull(bb)) > iou_threshold:
            # 当前切片截断obb框后仍为4个点，加入到有效obb框的坐标中
            if len(intersection[1]) == 4:
                new_bb.append(np.reshape(cv2.convexHull(intersection[1]), [8]))
                new_idx.append(idx)
            # 当前切片截断obb框后多于4个点，求交集的最小外接矩，该最小外接矩即为该切片中的obb框
            elif len(intersection[1]) > 4:
                new_bb.append(np.reshape(cv2.boxPoints(cv2.minAreaRect(intersection[1])), [8]))
                new_idx.append(idx)
            else:
                continue

    return np.array(new_bb).astype(np.int32), new_idx


def convert_absolute_to_relative(effective_obb, effective_idx, cls_names, s_w, s_h):
    """
    把在大图中的绝对坐标转换为patch中的相对坐标
    :param effective_obb: 该patch中的有效obb坐标
    :param effective_idx: 
    :param s_w: 
    :param s_h: 
    :return: 
    """
    relative_coor_obb = []
    obj_label = []

    # 大图坐标转成切片中的绝对坐标
    for i in range(len(effective_obb)):
        obb_list = effective_obb[i].tolist()
        label = cls_names[effective_idx[i]]
        for j in range(0, 8, 2):
            obb_list[j] = obb_list[j] - s_w
            obb_list[j + 1] = obb_list[j + 1] - s_h
        relative_coor_obb.append(obb_list)
        obj_label.append(label)

    return relative_coor_obb, obj_label


def save_to_aircas_xml(xml_path, relative_coor_obb, obj_label, im_width, im_height, band_num, crop_name):
    """
    保存为AIRCAS格式的xml
    :param xml_path: 
    :param relative_coor_obb: obb相对于切片的标注坐标
    :param obj_label: 
    :param im_width: 
    :param im_height: 
    :param band_num: 
    :param crop_name: 
    :return: 
    """
    doc = Document()
    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)
    # source
    source = doc.createElement("source")
    filename = doc.createElement("filename")
    filename.appendChild(doc.createTextNode(crop_name))
    source.appendChild(filename)
    origin = doc.createElement("origin")
    origin.appendChild(doc.createTextNode("GF2/GF3"))
    source.appendChild(origin)
    annotation.appendChild(source)
    # research
    research = doc.createElement("research")
    version = doc.createElement("version")
    version.appendChild(doc.createTextNode(str(1.0)))
    research.appendChild(version)
    provider = doc.createElement("provider")
    provider.appendChild(doc.createTextNode("FAIR1M"))
    research.appendChild(provider)
    author = doc.createElement("author")
    author.appendChild(doc.createTextNode("Cyber"))
    research.appendChild(author)
    pluginname = doc.createElement("pluginname")
    pluginname.appendChild(doc.createTextNode("FAIR1M"))
    research.appendChild(pluginname)
    pluginclass = doc.createElement("pluginclass")
    pluginclass.appendChild(doc.createTextNode("object detection"))
    research.appendChild(pluginclass)
    time = doc.createElement("time")
    time.appendChild(doc.createTextNode("2021-07-21"))
    research.appendChild(time)
    annotation.appendChild(research)
    # size
    size = doc.createElement("size")
    width = doc.createElement("width")
    width.appendChild(doc.createTextNode(str(im_width)))
    size.appendChild(width)
    height = doc.createElement("height")
    height.appendChild(doc.createTextNode(str(im_height)))
    size.appendChild(height)
    depth = doc.createElement("depth")
    depth.appendChild(doc.createTextNode(str(band_num)))
    size.appendChild(depth)
    annotation.appendChild(size)

    objects = doc.createElement("objects")
    for (obb, label) in zip(relative_coor_obb, obj_label):
        nodeobject = doc.createElement("object")
        # coordinate
        coordinate = doc.createElement("coordinate")
        coordinate.appendChild(doc.createTextNode("pixel"))
        nodeobject.appendChild(coordinate)
        # type
        type = doc.createElement("type")
        type.appendChild(doc.createTextNode("rectangle"))
        nodeobject.appendChild(type)
        # description
        description = doc.createElement("description")
        description.appendChild(doc.createTextNode("None"))
        nodeobject.appendChild(description)
        # possibleresult
        possibleresult = doc.createElement("possibleresult")
        nodename = doc.createElement("name")
        nodename.appendChild(doc.createTextNode(label))
        possibleresult.appendChild(nodename)
        nodeobject.appendChild(possibleresult)
        # points
        points = doc.createElement("points")
        # 旋转框坐标
        point = doc.createElement("point")
        point.appendChild(doc.createTextNode("{:.6f},{:.6f}".format(obb[0], obb[1])))
        points.appendChild(point)
        # 旋转框坐标
        point = doc.createElement("point")
        point.appendChild(doc.createTextNode("{:.6f},{:.6f}".format(obb[2], obb[3])))
        points.appendChild(point)
        # 旋转框坐标
        point = doc.createElement("point")
        point.appendChild(doc.createTextNode("{:.6f},{:.6f}".format(obb[4], obb[5])))
        points.appendChild(point)
        # 旋转框坐标
        point = doc.createElement("point")
        point.appendChild(doc.createTextNode("{:.6f},{:.6f}".format(obb[6], obb[7])))
        points.appendChild(point)
        # 旋转框坐标
        point = doc.createElement("point")
        point.appendChild(doc.createTextNode("{:.6f},{:.6f}".format(obb[0], obb[1])))
        points.appendChild(point)
        nodeobject.appendChild(points)
        # 加入到objects节中
        objects.appendChild(nodeobject)

    annotation.appendChild(objects)

    # 将标注信息写入到xml中
    fp = open(xml_path, "w")
    doc.writexml(fp, indent="\n", addindent="  ", encoding="utf-8")
    fp.close()


def save_to_disk(patch_data, crop_root, crop_name, coor, relative_coor_obb, obj_label, band_num):
    """
    被裁剪的切片与标注落盘
    :param patch_data: 
    :param crop_root: 
    :param crop_name: 
    :param coor: 被裁剪切片在大图中的坐标
    :param relative_coor_obb: obb框相对于patch的坐标
    :param obj_label: obb框的类别
    :param band_num: 波段数
    :return: 
    """
    s_w, s_h, e_w, e_h = coor

    # 保存切片png
    check_path(os.path.join(os.path.join(crop_root, "imgs")))
    png_path = os.path.join(os.path.join(crop_root, "imgs", crop_name + ".png"))
    img = Image.fromarray(patch_data.transpose(1, 2, 0))
    img.save(png_path)

    # 保存切片xml
    check_path(os.path.join(os.path.join(crop_root, "xmls")))
    xml_path = os.path.join(os.path.join(crop_root, "xmls", crop_name + ".xml"))
    save_to_aircas_xml(xml_path, relative_coor_obb, obj_label, e_w - s_w, e_h - s_h, band_num, crop_name + ".png")


def patch_visible(crop_root, crop_name, patch_data, relative_coor_obb, bgr):
    """
    将裁剪后的OBB框画到切片上
    :param crop_root: 
    :param crop_name: 
    :param patch_data: 
    :param relative_coor_obb: OBB框在切片中的坐标
    :param bgr: 
    :return: 
    """
    check_path(os.path.join(os.path.join(crop_root, "vis")))
    img = patch_data.transpose(1, 2, 0)[:, :, 0:3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pts = np.array(relative_coor_obb).reshape([-1, 4, 2])
    cv2.polylines(img, pts, isClosed=True, color=bgr, thickness=2)
    cv2.imwrite(os.path.join(crop_root, "vis", crop_name + "_vis.png"), img)


def sliding_window_crop(tif_path, crop_root, rectangle_obb, cls_names, vis_flag, bgr,
                        crop_w, crop_h, stride_w, stride_h, iou_threshold=0.5):
    """
    对obb标注进行滑窗裁剪
    :param tif_path: 
    :param crop_root: 
    :param rectangle_obb: 
    :param cls_names: 
    :param crop_w: 
    :param crop_h: 
    :param stride_w: 
    :param stride_h: 
    :param iou_threshold: IoU阈值
    :return: 
    """
    tif_name = os.path.basename(tif_path).split(".")[0]
    crop_root = os.path.join(crop_root, tif_name)
    im_proj, im_geotrans, im_width, im_height, band_num, dataset = get_tif_information(tif_path)

    # 目前只支持对Byte和UInt16类型的图像进行处理，其它类型遥感影像暂不支持
    band = dataset.GetRasterBand(1)
    typeName = gdal.GetDataTypeName(band.DataType)
    if typeName == "Byte":
        dataType = gdalconst.GDT_Byte
    elif typeName == "UInt16":
        dataType = gdalconst.GDT_UInt16
    else:
        print("Only Byte and UInt16 is supported, while given image is %s!" % typeName)
        return False

    # 记录该大图中目标坐标的范围
    obj_range = np.array(rectangle_obb).reshape([-1, 4, 2])     # [[[x, y], [x, y], ...], [[x, y], [x, y], ...]]
    obj_range = [np.min(obj_range[:, :, 0]), np.min(obj_range[:, :, 1]),
                 np.max(obj_range[:, :, 0]), np.max(obj_range[:, :, 1])]

    # 记录大图的宽高
    min_height, min_width = 0, 0
    max_height, max_width = im_height, im_width
    height_range, width_range = max_height - min_height, max_width - min_width
    # 计算各个方向需要多少滑窗
    num_h = int(np.ceil(float(height_range - crop_h) / stride_h) + 1)    # h方向需要多少滑窗
    num_w = int(np.ceil(float(width_range - crop_w) / stride_w) + 1)     # w方向需要多少滑窗

    i = 0
    print("{}宽高为：({}, {})，被裁切为{}*{}块".format(tif_name, width_range, height_range, num_h, num_w))
    for index_h in range(0, num_h):
        for index_w in range(0, num_w):
            i += 1
            s_h = min_height + index_h * stride_h   # 滑窗h方向起始位置
            e_h = min(s_h + crop_h, max_height)     # 滑窗h方向结束位置
            s_h = e_h - crop_h
            s_w = min_width + index_w * stride_w    # 滑窗w方向起始位置
            e_w = min(s_w + crop_w, max_width)      # 滑窗w方向结束位置
            s_w = e_w - crop_w

            # 判断该切片中是否包含目标，iou_threshold是像素面积，通常设置为0.5
            # 小于iou_threshold表示该切片与所有obb框无交集；大于iou_threshold表示会有至少1个像素的交集
            # 但后续会通过IoU阈值滤除在该切片中过小的obb框
            patch_coor = [s_w, s_h, e_w, e_h]
            if is_bboxes_in_patch(obj_range, patch_coor) < iou_threshold:
                continue

            # 提取有效目标框，如果被截断，则进行补全
            effective_obb, effective_idx = select_bbox(rectangle_obb, patch_coor, iou_threshold)

            # 大图坐标转成切片中的绝对坐标
            if len(effective_obb) == 0:
                continue
            relative_coor_obb, obj_label = \
                convert_absolute_to_relative(effective_obb, effective_idx, cls_names, s_w, s_h)

            # 保存为AIRCAS格式的训练集
            crop_name = "{}_{}_{}_{}_{}".format(tif_name, s_w, s_h, e_w, e_h)
            # 遥感影像存在黑边，但目标均在非黑边区域中，因此上述步骤可以去除大部分无目标的黑边切片，之后就无须再判断黑边区域
            patch_data = dataset.ReadAsArray(s_w, s_h, e_w - s_w, e_h - s_h)
            coor = (s_w, s_h, e_w, e_h)
            # 被裁剪的切片与标注落盘
            save_to_disk(patch_data, crop_root, crop_name, coor, relative_coor_obb, obj_label, band_num)

            print("\t处理完毕第{}个切片，切片左上右下坐标为：({}, {}), ({}, {})".format(i, s_w, s_h, e_w, e_h))

            # 切片可视化
            if vis_flag:
                patch_visible(crop_root, crop_name, patch_data, relative_coor_obb, bgr)

    print("总共生成{}个切片".format(i))


def parser():
    """
    解析传入参数
    :return: 
    """
    parser = argparse.ArgumentParser(description="obb crop images")
    # 原始标注路径
    parser.add_argument("--tif_root", type=str,
                        default="/mnt/sdb/share1416/airalgorithm/datasets/AnnoDataDet/Origin/imgs/")
    parser.add_argument("--xml_root", type=str,
                        default="/mnt/sdb/share1416/airalgorithm/code/OrientedObjectDetection/XML1")
    # 切片保存路径
    parser.add_argument("--crop_root", type=str,
                        default="/mnt/sdb/share1416/airalgorithm/code/OrientedObjectDetection/patches")
    # 切片可视化
    parser.add_argument("--vis_flag", type=bool, default=True)
    parser.add_argument("--bgr", type=tuple, default=(0, 0, 255))
    # 滑窗参数
    parser.add_argument("--crop_h", type=int, default=10000)
    parser.add_argument("--crop_w", type=int, default=10000)
    parser.add_argument("--stride_rate", type=float, default=1)
    parser.add_argument("--iou_threshold", type=float, default=0.5)

    args = parser.parse_args()

    return args


def main():
    args = parser()

    tif_root = args.tif_root
    xml_root = args.xml_root
    crop_root = args.crop_root
    vis_flag = args.vis_flag
    bgr = args.bgr
    crop_h = args.crop_h
    crop_w = args.crop_w
    stride_rate = args.stride_rate
    iou_threshold = args.iou_threshold

    # 计算滑窗步长
    check_path(crop_root)
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))

    # 遍历所有影像
    for tif_file in os.listdir(tif_root):
        if "Beijing" not in tif_file:
            continue
        tif_path = os.path.join(tif_root, tif_file)
        # xml_path = os.path.join(xml_root, tif_file.replace(".tif", ".xml"))
        # 星图标注数据后缀为".yaml"
        xml_path = os.path.join(xml_root, tif_file.replace(".tif", ".xml"))

        # 读取原始xml中的坐标框
        xml_info = read_aircas_xml(xml_path)
        raw_obb, cls_names = xml_info["bboxes"], xml_info["cls_names"]

        # 判断哪些标注是非矩形框
        non_rectangle_idx = select_non_rectangle(raw_obb)
        # 根据索引，将非矩形目标框转换为最小外接矩目标框
        rectangle_obb = non_rectangle_to_rectangle(raw_obb, non_rectangle_idx)

        # 滑窗裁剪
        sliding_window_crop(tif_path, crop_root, rectangle_obb, cls_names, vis_flag, bgr,
                            crop_w, crop_h, stride_w, stride_h, iou_threshold)


if __name__ == "__main__":
    main()
    print("Done!")
