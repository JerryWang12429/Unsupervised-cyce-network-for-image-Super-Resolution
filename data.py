import os
import sys
import numpy as np
import cv2
from glob import glob
from PIL import Image
import PIL
import pdb

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.utils.data as data
import torch.nn.functional as nnF
import torch

from custom_transform import NRandomCrop

High_Data = ["/media/pria/data/YuLi/Degradation/data/Train/HR"]
Low_Data = ["/media/pria/data/YuLi/Degradation/data/Train/LR"]


class get_data(Dataset):
    def __init__(self, data_hr, data_lr):
        self.hr_imgs = [os.path.join(d, i) for d in data_hr for i in os.listdir(d) if os.path.isfile(os.path.join(d, i))]
        self.lr_imgs = [os.path.join(d, i) for d in data_lr for i in os.listdir(d) if os.path.isfile(os.path.join(d, i))]
        self.lr_len = len(self.lr_imgs)
        self.lr_shuf = np.arange(self.lr_len)
        np.random.shuffle(self.lr_shuf)
        self.lr_idx = 0

        # mean, sd = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        # self.preproc_64 = transforms.Compose(
        #     [NRandomCrop(size=64, n=5, padding=1),
        #      transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean, sd)(transforms.ToTensor()(crop)) for crop in crops]))])
        # self.preproc_16 = transforms.Compose(
        #     [NRandomCrop(size=16, n=5, padding=1),
        #      transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean, sd)(transforms.ToTensor()(crop)) for crop in crops]))])
        self.preproc_64 = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.preproc_16 = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.hr_imgs)

    def __getitem__(self, index):
        data = {}
        hr = cv2.imread(self.hr_imgs[index])
        lr = cv2.imread(self.lr_imgs[self.lr_shuf[self.lr_idx]])
        self.lr_idx += 1
        if self.lr_idx >= self.lr_len:
            self.lr_idx = 0
            np.random.shuffle(self.lr_shuf)
        data["z"] = torch.randn(1, 64, dtype=torch.float32)
        data["lr"] = self.preproc_16(lr)
        data["hr"] = self.preproc_64(hr)
        data["hr"] = torch.unsqueeze(data["hr"], 0)
        data["hr_down"] = nnF.interpolate(data["hr"], scale_factor=0.25, mode='bicubic', align_corners=True)
        data["hr"] = torch.squeeze(data["hr"], 0)
        data["hr_down"] = torch.squeeze(data["hr_down"], 0)
        data["hr_down"].clamp(min=0, max=255)
        return data

    def get_noise(self, n):
        return torch.randn(n, 1, 64, dtype=torch.float32)


def get_loader(dataname, bs=1):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = test_loader(dataname, transform)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=bs,
                             shuffle=False, num_workers=2, pin_memory=True)
    return data_loader


class test_loader(data.Dataset):
    def __init__(self, datasets, transform):
        assert datasets, print('no datasets specified')
        self.transform = transform
        self.img_list = []
        dataset = datasets
        if dataset == 'testset':
            img_path = '/media/pria/data/YuLi/Degradation/data/Set14/LR_bicubic/X4'
            list_name = (glob(os.path.join(img_path, "*.png")))
            list_name.sort()
            for filename in list_name:
                self.img_list.append(filename)
        else:
            img_path = dataset
            list_name = (glob(os.path.join(img_path, "*.png")))
            list_name.sort()
            for filename in list_name:
                self.img_list.append(filename)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data = {}
        inp16 = Image.open(self.img_list[index]).convert('RGB')
        width, height = inp16.size
        inp64 = inp16.resize((width * 4, height * 4), resample=PIL.Image.BICUBIC)
        data['img64'] = self.transform(inp64)
        data['img16'] = self.transform(inp16)
        data['imgpath'] = self.img_list[index]
        return data


def get_loader_HR_test(dataname, bs=1):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.RandomCrop(64),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = test_loader(dataname, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    return data_loader
