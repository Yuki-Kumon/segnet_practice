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
train_data_path = os.path.join(cwd, 'data/CamVid/train/')
test_data_path = os.path.join(cwd, 'data/CamVid/test/')
val_data_path = os.path.join(cwd, 'data/CamVid/val/')
train_label_path = os.path.join(cwd, 'data/CamVid/trainannot/')
test_label_path = os.path.join(cwd, 'data/CamVid/testannot/')
val_label_path = os.path.join(cwd, 'data/CamVid/valannot/')


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
            label = self.transform(label)
        return image, label


class SegNet(nn.Module):
    '''
    SegNet Model
    '''
    def __init__(self, input_number, label_number, dropout_ratio):
        # initialization of class
        super(SegNet, self).__init__()

        # define the convolution layer of encoder
        self.conv1_1 = nn.Conv2d(input_number, 64, kernel_size=3, stride=1, padding=1)
        self.bachnorm1_1 = nn.BatchNorm2d(num_features=64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bachnorm1_2 = nn.BatchNorm2d(num_features=64)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bachnorm2_1 = nn.BatchNorm2d(num_features=128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bachnorm2_2 = nn.BatchNorm2d(num_features=128)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bachnorm3_1 = nn.BatchNorm2d(num_features=256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bachnorm3_2 = nn.BatchNorm2d(num_features=256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bachnorm3_3 = nn.BatchNorm2d(num_features=256)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bachnorm4_1 = nn.BatchNorm2d(num_features=512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bachnorm4_2 = nn.BatchNorm2d(num_features=512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bachnorm4_3 = nn.BatchNorm2d(num_features=512)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bachnorm5_1 = nn.BatchNorm2d(num_features=512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bachnorm5_2 = nn.BatchNorm2d(num_features=512)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bachnorm5_3 = nn.BatchNorm2d(num_features=512)

        # define the convolution layer of decoder
        self.conv5_3_d = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bachnorm5_3_d = nn.BatchNorm2d(num_features=512)
        self.conv5_2_d = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bachnorm5_2_d = nn.BatchNorm2d(num_features=512)
        self.conv5_1_d = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bachnorm5_1_d = nn.BatchNorm2d(num_features=512)

        self.conv4_3_d = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bachnorm4_3_d = nn.BatchNorm2d(num_features=512)
        self.conv4_2_d = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bachnorm4_2_d = nn.BatchNorm2d(num_features=512)
        self.conv4_1_d = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bachnorm4_1_d = nn.BatchNorm2d(num_features=256)

        self.conv3_3_d = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bachnorm3_3_d = nn.BatchNorm2d(num_features=256)
        self.conv3_2_d = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bachnorm3_2_d = nn.BatchNorm2d(num_features=256)
        self.conv3_1_d = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bachnorm3_1_d = nn.BatchNorm2d(num_features=128)

        self.conv2_2_d = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bachnorm2_2_d = nn.BatchNorm2d(num_features=128)
        self.conv2_1_d = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bachnorm2_1_d = nn.BatchNorm2d(num_features=64)

        self.conv1_2_d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bachnorm1_2_d = nn.BatchNorm2d(num_features=64)
        self.conv1_1_d = nn.Conv2d(64, label_number, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout2d(p=dropout_ratio)

    def forward(self, x):
        # define the forward network
        # Encoder
        x1_1 = F.relu(self.bachnorm1_1(self.conv1_1(x)))
        x1_2 = F.relu(self.bachnorm1_2(self.conv1_2(x1_1)))
        x1p, id1 = F.max_pool2d(self.dropout(x1_2), kernel_size=2, stride=2, return_indices=True)

        x2_1 = F.relu(self.bachnorm2_1(self.conv2_1(x1p)))
        x2_2 = F.relu(self.bachnorm2_2(self.conv2_2(x2_1)))
        x2p, id2 = F.max_pool2d(self.dropout(x2_2), kernel_size=2, stride=2, return_indices=True)

        x3_1 = F.relu(self.bachnorm3_1(self.conv3_1(x2p)))
        x3_2 = F.relu(self.bachnorm3_2(self.conv3_2(x3_1)))
        x3_3 = F.relu(self.bachnorm3_3(self.conv3_3(x3_2)))
        x3p, id3 = F.max_pool2d(self.dropout(x3_3), kernel_size=2, stride=2, return_indices=True)

        x4_1 = F.relu(self.bachnorm4_1(self.conv4_1(x3p)))
        x4_2 = F.relu(self.bachnorm4_2(self.conv4_2(x4_1)))
        x4_3 = F.relu(self.bachnorm4_3(self.conv4_3(x4_2)))
        x4p, id4 = F.max_pool2d(self.dropout(x4_3), kernel_size=2, stride=2, return_indices=True)

        x5_1 = F.relu(self.bachnorm5_1(self.conv5_1(x4p)))
        x5_2 = F.relu(self.bachnorm5_2(self.conv5_2(x5_1)))
        x5_3 = F.relu(self.bachnorm5_3(self.conv5_3(x5_2)))
        x5p, id5 = F.max_pool2d(self.dropout(x5_3), kernel_size=2, stride=2, return_indices=True)

        # Decoder
        x5d = F.max_unpool2d(self.dropout(x5p), id5, kernel_size=2, stride=2)
        x5_3_d = F.relu(self.bachnorm5_3_d(self.conv5_3_d(x5d)))
        x5_2_d = F.relu(self.bachnorm5_2_d(self.conv5_2_d(x5_3_d)))
        x5_1_d = F.relu(self.bachnorm5_1_d(self.conv5_1_d(x5_2_d)))

        x4d = F.max_unpool2d(self.dropout(x5_1_d), id4, kernel_size=2, stride=2)
        x4_3_d = F.relu(self.bachnorm4_3_d(self.conv4_3_d(x4d)))
        x4_2_d = F.relu(self.bachnorm4_2_d(self.conv4_2_d(x4_3_d)))
        x4_1_d = F.relu(self.bachnorm4_1_d(self.conv4_1_d(x4_2_d)))

        x3d = F.max_unpool2d(self.dropout(x4_1_d), id3, kernel_size=2, stride=2)
        x3_3_d = F.relu(self.bachnorm3_3_d(self.conv3_3_d(x3d)))
        x3_2_d = F.relu(self.bachnorm3_2_d(self.conv3_2_d(x3_3_d)))
        x3_1_d = F.relu(self.bachnorm3_1_d(self.conv3_1_d(x3_2_d)))

        x2d = F.max_unpool2d(self.dropout(x3_1_d), id2, kernel_size=2, stride=2)
        x2_2_d = F.relu(self.bachnorm2_2_d(self.conv2_2_d(x2d)))
        x2_1_d = F.relu(self.bachnorm2_1_d(self.conv2_1_d(x2_2_d)))

        x1d = F.max_unpool2d(self.dropout(x2_1_d), id1, kernel_size=2, stride=2)
        x1_2_d = F.relu(self.bachnorm1_2_d(self.conv1_2_d(x1d)))
        x1_1_d = self.conv1_1_d(x1_2_d)

        return x1_1_d


# create my dataset
trans = transforms.Compose([
    transforms.Resize((480, 360)),
    transforms.ToTensor()
])
train_data_set = MyDataset(train_data_path, train_label_path, trans)
test_data_set = MyDataset(test_data_path, test_label_path, trans)
val_data_set = MyDataset(val_data_path, val_label_path, trans)

# create my dataloader
train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=64, shuffle=True)

print("Complete the preparation of dataset")

# create SegNet Model
# input cahnnel number is 3
# label number is 12
model = SegNet(3, 12, 0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

print("Complete the preparation of model")

# if needed, load pretrained model
if(0):
    PATH = os.path.join(cwd, 'model')
    model.load_state_dict(torch.load(PATH))
    print("Loaded the pretrained model")


# define train function
def train(epoch):
    '''
    train function
    '''
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        # Variable型への変換(統合されたので省略)
        # image, label = Variable(image), Variable(label)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(image), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))


"""
print(train_data_set[12][0].numpy())
test = train_data_set[12][1].numpy()
plt.imshow(test[0])
plt.show()
"""
