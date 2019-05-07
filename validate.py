# -*- coding: utf-8 -*-

"""
Validate model by validation data
Author :
    Yuki Kumon
Last Update :
    2019-05-05
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
from main import SegNet, MyDataset, trans1, trans2  # load SegNet model and dataset and transform


# for running in Google Colab
"""
if(1):
    from google.colab import drive
    drive.mount('/content/drive')
    # 必要ならば以下のようにディレクトリ移動する
    %cd /content/drive/'My Drive'/'Colab'/
"""


# set file path
cwd = os.getcwd()
val_data_path = os.path.join(cwd, 'data/CamVid/val/')
val_label_path = os.path.join(cwd, 'data/CamVid/valannot/')


def writeImage(image_tensor, filename):
    """ label data to colored image """
    image = image_tensor.detach().numpy()
    Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Pole = [192, 192, 128]
    Road_marking = [255, 69, 0]
    Road = [128, 64, 128]
    Pavement = [60, 40, 222]
    Tree = [128, 128, 0]
    SignSymbol = [192, 128, 128]
    Fence = [64, 64, 128]
    Car = [64, 0, 128]
    Pedestrian = [64, 64, 0]
    Bicyclist = [0, 128, 192]
    Unlabelled = [0, 0, 0]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Sky, Building, Pole, Road_marking, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    for l in range(0, 12):
        r[image == l] = label_colours[l, 0]
        g[image == l] = label_colours[l, 1]
        b[image == l] = label_colours[l, 2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:, :, 0] = r / 1.0
    rgb[:, :, 1] = g / 1.0
    rgb[:, :, 2] = b / 1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)


# create My Model
model = SegNet(3, 12, 0.01)
model = model.to('cuda')
PATH = os.path.join(cwd, 'model')
model.load_state_dict(torch.load(PATH))
# PATH_txt = os.path.join(cwd, 'epoch_num.txt')
print("Loaded the pretrained model")

# create validation dataset
val_data_set = MyDataset(val_data_path, val_label_path, trans1, trans2)
val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=1, shuffle=True)


# define validate function
def val():
    '''
    validate function
    '''
    model.eval()
    for (image, label) in val_loader:
        output = model(image)
        writeImage(output, os.path.join(cwd, 'output/im_by_model.png'))
        writeImage(label, os.path.join(cwd, 'output/im_by_label.png'))


if __name__ == '__main__':
    val()
