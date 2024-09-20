import os
import shutil
txt_path="./ym_small.txt"
files=os.listdir("../imgs")
with open(txt_path, "r") as f:
    for line in f.readlines():
        line = line.strip()
        print(line)
        for file in files:
            if line in file:
                print("youyouyou")
                print(file)
                src_path="../imgs/"+file
                shutil.move(src_path,"./"+file)

