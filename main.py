import os
import sys
import numpy as np
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
from torchsummary import summary

from model import High2Low, Discriminator, Low2High
from data import get_data, High_Data, Low_Data, get_loader, get_loader_HR_test
from Loss import TVLoss

from torchviz import make_dot

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--gpu", action="store", dest="gpu", default=0, help="using GPU number")
parser.add_argument("-e", "--epoch", action="store", dest="epoch", help="Training epoch", required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    seed_num = 2020
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loss_D_h2l_log = []
    loss_D_l2h_log = []
    loss_G_h2l_log = []
    loss_G_l2h_log = []
    loss_cycle_log = []

    max_epoch = int(args.epoch)
    learn_rate = 1e-4

    G_h2l = High2Low().cuda()
    D_h2l = Discriminator(16).cuda()
    G_l2h = Low2High().cuda()
    D_l2h = Discriminator(64).cuda()
    mse = nn.MSELoss()
    TVloss = TVLoss()
    L1 = nn.L1Loss()
    upsample = nn.Upsample(scale_factor=4, mode='bicubic')

    optim_D_h2l = optim.Adam(filter(lambda p: p.requires_grad, D_h2l.parameters()), lr=learn_rate, betas=(0.0, 0.9))
    optim_G_h2l = optim.Adam(G_h2l.parameters(), lr=learn_rate, betas=(0.0, 0.9))
    optim_D_l2h = optim.Adam(filter(lambda p: p.requires_grad, D_l2h.parameters()), lr=learn_rate, betas=(0.0, 0.9))
    optim_G_l2h = optim.Adam(G_l2h.parameters(), lr=learn_rate, betas=(0.0, 0.9))

    data = get_data(High_Data, Low_Data)
    loader = DataLoader(dataset=data, batch_size=32, shuffle=True)
    test_loader = get_loader("testset", bs=1)
    num_test = 14
    test_save = "intermid_results"
    if not os.path.isdir(test_save):
        os.makedirs(test_save)
        os.mkdir(test_save + "/imgs")
        os.mkdir(test_save + "/imgs_gen")
        os.mkdir(test_save + "/models")

    print("Training High-to-Low and Low-to-High...")
    for ep in tqdm(range(1, max_epoch + 1)):
        G_h2l.train()
        D_h2l.train()
        G_l2h.train()
        D_l2h.train()
        for i, batch in enumerate(loader):
            optim_D_h2l.zero_grad()
            optim_D_l2h.zero_grad()
            optim_G_h2l.zero_grad()
            optim_G_l2h.zero_grad()

            zs = batch["z"].cuda()
            lrs = batch["lr"].cuda()
            hrs = batch["hr"].cuda()
            downs = batch["hr_down"].cuda()

            lr_gen = G_h2l(hrs, zs)
            lr_gen_detach = lr_gen.detach()
            hr_gen = G_l2h(lr_gen_detach)
            hr_gen_detach = hr_gen.detach()

            # update discriminator
            loss_D_h2l = nn.ReLU()(1.0 - D_h2l(lrs)).mean() + nn.ReLU()(1 + D_h2l(lr_gen_detach)).mean()
            loss_D_l2h = nn.ReLU()(1.0 - D_l2h(hrs)).mean() + nn.ReLU()(1 + D_l2h(hr_gen_detach)).mean()
            loss_D_h2l.backward()
            loss_D_l2h.backward()
            optim_D_h2l.step()
            optim_D_l2h.step()

            # update generator
            optim_D_h2l.zero_grad()
            gan_loss_h2l = -D_h2l(lr_gen).mean()
            mse_loss_h2l = L1(lr_gen, downs)
            hr_upsample = upsample(lrs)
            lr_gen_up = G_h2l(hr_upsample, zs)
            mse_loss_h2l_up = L1(lr_gen_up, lrs)

            loss_G_h2l = 0.5 * mse_loss_h2l + 1 * gan_loss_h2l + 0.5 * mse_loss_h2l_up
            loss_G_h2l.backward()
            optim_G_h2l.step()

            optim_D_l2h.zero_grad()
            gan_loss_l2h = -D_l2h(hr_gen).mean()
            mse_loss_l2h = L1(hr_gen, hrs)
            hr_gen_bicubic = G_l2h(downs)
            mse_loss_l2h_bicubic = L1(hr_gen, hr_gen_bicubic)

            loss_G_l2h = 0.5 * mse_loss_l2h + 0.5 * mse_loss_l2h_bicubic + 1 * gan_loss_l2h + 0.001 * TVloss(hr_gen)
            loss_G_l2h.backward()
            optim_G_l2h.step()

            # cycle construct training
            optim_G_h2l.zero_grad()
            optim_G_l2h.zero_grad()

            hr_gen_cyc = G_l2h(lrs)
            lr_gen_cyc = G_h2l(hr_gen_cyc, zs)

            loss_cycle = L1(lr_gen_cyc, lrs)
            loss_cycle.backward()
            optim_G_h2l.step()
            optim_G_l2h.step()

            print(" {}({}) D_h2l: {:.3f}, D_l2h: {:.3f}, G_h2l: {:.3f}, G_l2h: {:.3f}, cycle_loss: {:.3f} \r".format(
                i + 1, ep, loss_D_h2l.item(), loss_D_l2h.item(), loss_G_h2l.item(), loss_G_l2h.item(), loss_cycle.item()),
                end=" ")

            loss_D_h2l_log.append(loss_D_h2l.item())
            loss_D_l2h_log.append(loss_D_l2h.item())
            loss_G_h2l_log.append(loss_G_h2l.item())
            loss_G_l2h_log.append(loss_G_l2h.item())
            loss_cycle_log.append(loss_cycle.item())

            loss_all = (loss_D_h2l_log, loss_D_l2h_log, loss_G_h2l_log, loss_G_l2h_log, loss_cycle_log)

            file = open("loss_log.txt", "w")
            json.dump(loss_all, file)
            file.close()

        print("\n Testing and saving...")
        G_h2l.eval()
        D_h2l.eval()
        G_l2h.eval()
        D_l2h.eval()

        if ep % 10 == 0:
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
                cv2.imwrite("{}/imgs/{}_{}_lr.png".format(test_save, ep, i + 1), np_low)
                cv2.imwrite("{}/imgs/{}_{}_sr.png".format(test_save, ep, i + 1), np_gen)

            # HRim_loader = get_loader_HR_test("/media/pria/data/YuLi/CinCGAN-pytorch-master/code/DIV2K/Set14/HR", bs=1)
            # for i, sample in enumerate(HRim_loader):
            #     if i >= num_test:
            #         break
            #     low_temp = sample["img16"].numpy()
            #     low = torch.from_numpy(np.ascontiguousarray(low_temp[:, ::-1, :, :])).cuda()
            #     # print(low.shape)
            #     with torch.no_grad():
            #         z = torch.randn(1, 64, dtype=torch.float32).cuda()
            #         # print(z.shape)
            #         hign_gen = G_h2l(low, z)
            #     np_low = low.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
            #     np_gen = hign_gen.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
            #     np_low = (np_low - np_low.min()) / (np_low.max() - np_low.min())
            #     np_gen = (np_gen - np_gen.min()) / (np_gen.max() - np_gen.min())
            #     np_low = (np_low * 255).astype(np.uint8)
            #     np_gen = (np_gen * 255).astype(np.uint8)

            #     np_low = cv2.resize(np_low, (100, 100))
            #     np_gen = cv2.resize(np_gen, (100, 100))
            #     cv2.imwrite("{}/imgs_gen/{}_{}_GT.png".format(test_save, ep, i + 1), np_low)
            #     cv2.imwrite("{}/imgs_gen/{}_{}_lowgen.png".format(test_save, ep, i + 1), np_gen)

        save_file = "{}/models/model_epoch_{:03d}.pth".format(test_save, ep)
        torch.save({"G_h2l": G_h2l.state_dict(), "D_h2l": D_h2l.state_dict(),
                    "G_l2h": G_l2h.state_dict(), "D_l2h": D_l2h.state_dict()}, save_file)
        print("saved: ", save_file)

    print("finished.")
