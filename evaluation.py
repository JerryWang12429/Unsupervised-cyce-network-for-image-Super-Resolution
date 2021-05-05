import math
import os
from glob import glob
import cv2
import numpy as np
from natsort import natsorted


def psnr(original, contrast):
    mse = np.mean((original / 255. - contrast / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def eval_bicubic(f_nums, list_path1, list_path2):
    for i in range(0, f_nums):
        print("{},{}".format(list_path1[i], list_path2[i]))
        img_a = cv2.imread(list_path1[i])
        img_b = cv2.imread(list_path2[i])
        rows, cols, channels = img_a.shape
        img_b = cv2.resize(img_b, (cols, rows), interpolation=cv2.INTER_CUBIC)
        psnr_num = psnr(img_a, img_b)
        ssim_num = ssim(img_a, img_b)
        list_ssim.append(ssim_num)
        list_psnr.append(psnr_num)
    print("Average PSNR:", np.mean(list_psnr))
    print("Average SSIM:", np.mean(list_ssim))


if __name__ == '__main__':
    Path1 = '/media/pria/data/YuLi/Degradation/data/DIV2K_valid_HR'
    Path2 = '/media/pria/data/YuLi/Degradation/Degradation/test_results_0504/SR_copy'
    f_nums = len(os.listdir(Path1))
    list_psnr = []
    list_ssim = []

    print(f_nums)

    list_path1 = (glob(os.path.join(Path1, "*.png")))
    list_path2 = (glob(os.path.join(Path2, "*.png")))

    list_path1 = natsorted(list_path1)
    list_path2 = natsorted(list_path2)
    eval_bicubic(f_nums, list_path1, list_path2)
