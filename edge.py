import math
import os
from glob import glob
import cv2
import numpy as np
from natsort import natsorted


if __name__ == "__main__":
    Path1 = '/media/pria/data/YuLi/Degradation/data/Set14/LR_bicubic/X4'
    Path2 = '/media/pria/data/YuLi/Degradation/Degradation/SR_bicubic_0504'
    Path3 = '/media/pria/data/YuLi/Degradation/data/Set14/HR'

    f_nums = len(os.listdir(Path1))

    print(f_nums)

    list_path1 = (glob(os.path.join(Path1, "*.png")))
    list_path2 = (glob(os.path.join(Path2, "*.png")))
    list_path3 = (glob(os.path.join(Path3, "*.png")))

    list_path1 = natsorted(list_path1)
    list_path2 = natsorted(list_path2)
    list_path3 = natsorted(list_path3)

    test_save = "edge"
    if not os.path.isdir(test_save):
        os.makedirs(test_save)

    for i in range(0, f_nums):
        img_a = cv2.imread(list_path1[i])
        img_b = cv2.imread(list_path2[i])
        img_c = cv2.imread(list_path3[i])
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
        gray_c = cv2.cvtColor(img_c, cv2.COLOR_RGB2GRAY)
        kernel_size = 3
        blur_gray_a = cv2.GaussianBlur(gray_a, (kernel_size, kernel_size), 0)
        blur_gray_b = cv2.GaussianBlur(gray_b, (kernel_size, kernel_size), 0)
        blur_gray_c = cv2.GaussianBlur(gray_c, (kernel_size, kernel_size), 0)

        low_threshold = 1
        high_threshold = 10
        edges_a = cv2.Canny(blur_gray_a, low_threshold, high_threshold)
        edges_b = cv2.Canny(blur_gray_b, low_threshold, high_threshold)
        edges_c = cv2.Canny(blur_gray_c, low_threshold, high_threshold)

        cv2.imwrite("edge/{}_bicubic.png".format(i + 1), edges_a)
        cv2.imwrite("edge/{}_SR.png".format(i + 1), edges_b)
        cv2.imwrite("edge/{}_GT.png".format(i + 1), edges_c)
