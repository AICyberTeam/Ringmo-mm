import os
import shutil
import random
imgs = ['Beijing_Daxing', 'Chongqing', 'Fujian_fuzhou', 'Guangdong_Shenzhen', 'Guangdong_Zhanjiang', 'Guangzhou_jiangmen', 'Hainan_Haikou', 'Hainan_sanya', 'Jiangsu_nanjing', 'Liaoning_dalian']
for i in imgs:
    path="/mnt/sdb/share1416/airalgorithm/datasets/AnnoDataDet/patch_split_4000/"+i+"/xmls/imgset/val.txt"
    s_path="/mnt/sdb/share1416/airalgorithm/datasets/AnnoDataDet/patch_split_4000/"+i+"/xmls"
    d_path="/mnt/sdb/share1416/airalgorithm/datasets/AnnoDataDet/patch_split_4000/"+i+"/xmls/val"
    if not os.path.exists(d_path):
        os.mkdir(d_path)
    with open(path,"r") as f:
        for line in f:
            line=line.replace("\n","")
            shutil.copy(os.path.join(s_path,line+".xml"),os.path.join(d_path,line+".xml"))


# def split_file(xml_dir,output_dir_train,output_dir_val,ratio):
#     xml_files=os.listdir(dir)
#     file_num=len(xml_files)
#
#     split_point=int(file_num*ratio)
#     train_index=random.sample(xml_files,k=split_point)
#     for
#     for xml in train_index


