import os
import random

def split_data(base_path, ratio = 0.8):
    xml_path = base_path
    xml_files = [f.split(".")[0] for f in os.listdir(xml_path) if f.endswith(".xml")]

    #随机打乱
    random.shuffle(xml_files)

    #划分数据
    train_size = int(len(xml_files) * ratio)
    train_files = xml_files[:train_size]
    val_files = xml_files[train_size:]

    imgset_path = os.path.join(base_path,  "imgset")
    if not os.path.exists(imgset_path):
        os.mkdir(imgset_path)

    with open(os.path.join(imgset_path, "train.txt"), "w") as f:
        for item in train_files:
            f.write("%s\n" % item)

    with open(os.path.join(imgset_path, "val.txt"), "w") as f:
        for item in val_files:
            f.write("%s\n" % item)
    return train_files, val_files

# base_path = '/mnt/sdb/share1416/airalgorithm/datasets/InternalDatasets/'
imgs = ['Beijing_Daxing', 'Chongqing', 'Fujian_fuzhou', 'Guangdong_Shenzhen', 'Guangdong_Zhanjiang', 'Guangzhou_jiangmen', 'Hainan_Haikou', 'Hainan_sanya', 'Jiangsu_nanjing', 'Liaoning_dalian']
for i in imgs:
        # base_path = '/mnt/sdb/share1416/airalgorithm/datasets/InternalDatasets/'
        base_path = '/mnt/sdb/share1416/airalgorithm/datasets/AnnoDataDet/patch_split_4000/'+i+'/xmls'
        split_data(base_path)



