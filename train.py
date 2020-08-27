#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
import time
import numpy as np

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image


def readfile(path):
    # 按照传入的路径和txt文本参数，以只读形式打开
    fh = open(path, 'r')

    img_lines = []

    for line in fh:
        # 去除字符串首位字符
        line = line.strip('\n')
        # 去除字符串右边的字符
        line = line.rstrip('\n')
        # 分割字符串(空格)
        words = line.split()
        # words[0]是图片地址，words[1]是该图片的label
        img_lines.append((words[0], int(words[1])))

        return img_lines

'''
    x = np.zeros((len(img_lines), 28, 28, 1), dtype=np.uint8)
    y = np.zeros((len(img_lines)), dtype=np.uint8)

    for i, img_line in enumerate(img_lines):
        img = Image.open(img_line[0]).convert('L')
        x[i, :, :] = img.resize((28, 28))
        y[i] = img_line[1]

        return x, y
'''


def read_dataset(workspace_dir):
    # 分别将train set, validation set, testing set用readfile函数读进来
    print("Reading data... ...")
    # train_xx, train_yy = readfile(os.path.join(workspace_dir, "train.txt"))
    train_lines = readfile(os.path.join(workspace_dir, "train.txt"))
    print("Size of training data = {}".format(len(train_lines)))
    # val_xx, val_yy = readfile(os.path.join(workspace_dir, "valid.txt"))
    val_lines = readfile(os.path.join(workspace_dir, "valid.txt"))
    print("Size of validation data = {}".format(len(val_lines)))
    # test_xx, test_yy = readfile(os.path.join(workspace_dir, "test.txt"))
    test_lines = readfile(os.path.join(workspace_dir, "test.txt"))
    print("Size of testing data = {}".format(len(test_lines)))

    # return train_xx, train_yy, val_xx, val_yy, test_xx, test_yy
    return train_lines, val_lines, test_lines


class DigitDataset(Dataset):
    def __init__(self, lines, transform=None):
        super(DigitDataset, self).__init__()
        self.lines = lines
        self.transform = transform

    def __len__(self):
        # return len(self.x)
        return len(self.lines)

    def __getitem__(self, index):
        fn, label = self.lines[index]
        img = Image.open(fn).convert('L')
        # X = self.x[index]
        if self.transform is not None:
            # X = self.transform(X)
            img = self.transform(img)
        # Y = self.y[index]

        # return X, Y
        return img, label


def go_transform():
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop((28, 28)),
            transforms.ToTensor()
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop((28, 28)),
            transforms.ToTensor()
        ]
    )

    return train_transforms, test_transforms


class DigitModel(nn.Module):
    def __init__(self):
        super(DigitModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


def training(lr, num_epoch):
    my_model = DigitModel().cuda()

    # 分类任务使用交叉熵损失
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=lr)

    print("Start training... ...")
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        my_model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            train_pred = my_model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())
            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

        my_model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = my_model(data[0].cuda())
                batch_loss = loss(val_pred, data[1].cuda())

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()

        # Print结果
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' %
              (epoch + 1, num_epoch, time.time() - epoch_start_time,
               train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
               val_loss / val_set.__len__()))

    return my_model


def testing(my_model):
    my_model.eval()
    prediction = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = my_model(data[0].cuda())
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_label:
                prediction.append(y)

    return prediction


if __name__ == "__main__":
    learning_rate = 0.001
    epochs = 30
    batch_size = 128
    # train_x, train_y, val_x, val_y, test_x, test_y = read_dataset("/home/dzx/workSpace/Digit-recognition")
    train_lines, val_lines, test_lines = read_dataset("/home/dzx/workSpace/Digit-recognition")

    train_transform, test_transform = go_transform()

    train_set = DigitDataset(train_lines, train_transform)
    val_set = DigitDataset(val_lines, test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = training(learning_rate, epochs)

    test_set = DigitDataset(test_lines, test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    predict = testing(model)

    print(predict)
