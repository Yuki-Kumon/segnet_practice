# -*- coding: utf-8 -*-

"""
SegNet practice using street landscape pictures
Author :
    Yuki Kumon
Last Update :
    2019-05-02
"""


from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn  # ネットワーク構築用
import torch.optim as optim  # 最適化関数
import torch.nn.functional as F  # ネットワーク用の様々な関数
import torch.utils.data  # データセット読み込み関連
# from torch.utils.data.sampler import SubsetRandomSampler  # データセット分割
# from torch.autograd import Variable
import os
# import sys
import pandas as pd
# import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# set file path
cwd = os.getcwd()
# train_txt_path = os.path.join(cwd, 'data/CamVid/test.txt')
# test_txt_path = os.path.join(cwd, 'data/CamVid/test.txt')
train_data_path = os.path.join(cwd, 'data/CamVid/train/')
test_data_path = os.path.join(cwd, 'data/CamVid/test/')
train_label_path = os.path.join(cwd, 'data/CamVid/trainannot/')
test_label_path = os.path.join(cwd, 'data/CamVid/testannot/')


class MyDataset(Dataset):
    '''
    dataset class
    '''

    def __init__(self, image_path, label_path, transform=None):
        # set path
        self.image_path = image_path
        self.label_path = label_path
        # create file name list
        file_list = os.listdir(path=image_path)
        self.file_list = file_list
        # set transforms
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # load image and label
        image_name = self.file_list[idx]
        image = Image.open(os.path.join(self.image_path, image_name))
        label = Image.open(os.path.join(self.label_path, image_name))
        if self.transform:
            image = self.transform(image)
        return image, label


# create my dataset
train_data_set = MyDataset(train_data_path, train_label_path, transforms.Compose([
    transforms.Resize((480, 360)),
    transforms.ToTensor(),
    transforms.Normalize((5.0, 5.0, 5.0), (5.0, 5.0, 5.0))
]))

# create my dataloader
train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=64, shuffle=True)


print(train_data_set[12][0].size())
'''
test = train_data_set[12][0].numpy()
plt.imshow(test)
plt.show()
'''
