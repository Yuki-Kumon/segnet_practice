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


# set file path
cwd = os.getcwd()
train_txt_path = os.path.join(cwd, 'data/CamVid/test.txt')
test_txt_path = os.path.join(cwd, 'data/CamVid/test.txt')
train_data_path = os.path.join(cwd, 'data/CamVid/train')
test_data_path = os.path.join(cwd, 'data/CamVid/test')
train_label_path = os.path.join(cwd, 'data/CamVid/trainannot')
test_label_path = os.path.join(cwd, 'data/CamVid/testannot')

f = open(train_txt_path, 'r')
for line in f:
    print(os.path.basename(line.split(' ')[0]))
f.close()


class MyDataset(Dataset):
    '''
    dataset class
    '''

    def __init__(self, txt_path, data_path, label_path, transform=None):
        # set path
        self.data_path = data_path
        self.label_path = label_path
        # create file name list
        file_list = []
        f = open(txt_path, 'r')
        for line in f:
            file_list.append([os.path.basename(line.split(' ')[0]), os.path.basename(line.split(' ')[1])])
        f.close()
        self.file_list = file_list
        # set transforms
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # load image and label
        data_name = self.file_list[0, idx]
        label_name = self.file_list[1, idx]
        return image, label
