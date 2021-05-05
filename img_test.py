import os
import torch
import numpy as np
import cv2
import random
import argparse
import pdb

from model import Low2High
from data import get_loader

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--gpu", action="store", dest="gpu", default=0, help="using GPU number")

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    seed_num = 2020
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_loader = get_loader("/media/pria/data/YuLi/Degradation/data/DIV2K_valid_LR_mild/", bs=1)
    num_test = 100
    test_save = "test_results"
    if not os.path.isdir(test_save):
        os.makedirs(test_save)

    G_l2h = Low2High().cuda()
    G_l2h.load_state_dict(torch.load("intermid_results_0420/models/model_epoch_200.pth")["G_l2h"])
    G_l2h.eval()

    for i, sample in enumerate(test_loader):
        if i >= num_test:
            break
        low_temp = sample["img16"].numpy()
        low = torch.from_numpy(np.ascontiguousarray(low_temp[:, ::-1, :, :])).cuda()
        with torch.no_grad():
            hign_gen = G_l2h(low)
        np_low = low.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        np_gen = hign_gen.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        np_low = (np_low - np_low.min()) / (np_low.max() - np_low.min())
        np_gen = (np_gen - np_gen.min()) / (np_gen.max() - np_gen.min())
        np_low = (np_low * 255).astype(np.uint8)
        np_gen = (np_gen * 255).astype(np.uint8)
        cv2.imwrite("{}/{}_lr.png".format(test_save, i + 1), np_low)
        cv2.imwrite("{}/{}_sr.png".format(test_save, i + 1), np_gen)
