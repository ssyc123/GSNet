import os
import os.path
import shutil
from random import sample




if __name__ == '__main__':
    dir_path = r"/home/syc/mmdection/CG-Net-master/data/dota1-split-1024/test1024/images"
    tar_path = r"/home/syc/mmdection/CG-Net-master/data/dota1-split-1024/1"

    pic_list = os.listdir(dir_path)
    s_list = sample(pic_list, 100)
    for i in s_list:
        name = os.path.join(dir_path, i)
        shutil.copy(name, tar_path)

