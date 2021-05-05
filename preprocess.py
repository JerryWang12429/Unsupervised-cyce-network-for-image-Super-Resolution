import math
import os
from glob import glob
import cv2
import numpy as np

dir_path = '/media/pria/data/YuLi/Degradation/data/Train'


def random_crop(f_nums, list_name, crop_size, path):
    for i in range(0, f_nums):
        image = cv2.imread(list_path1[i])
        for number in range(0, 6):
            max_x = image.shape[1] - crop_size
            max_y = image.shape[0] - crop_size
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)
            crop = image[y: y + crop_size, x: x + crop_size]
            cv2.imwrite("{}/{}_{}_{}.png".format(path, i, number, crop_size), crop)


if __name__ == "__main__":

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        os.mkdir(dir_path + "/HR")
        os.mkdir(dir_path + "/LR")

    HRpath = dir_path + "/HR"
    LRpath = dir_path + "/LR"

    Path1 = '/media/pria/data/YuLi/Degradation/data/DIV2K_train_HR/'
    Path2 = '/media/pria/data/YuLi/Degradation/data/DIV2K_train_LR_mild'
    f_nums_1 = len(os.listdir(Path1))
    f_nums_2 = len(os.listdir(Path2))

    list_path1 = (glob(os.path.join(Path1, "*.png")))
    list_path2 = (glob(os.path.join(Path2, "*.png")))
    random_crop(f_nums_1, list_path1, 64, HRpath)
    random_crop(f_nums_2, list_path2, 16, LRpath)
