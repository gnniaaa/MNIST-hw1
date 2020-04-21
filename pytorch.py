#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time

BATCH_SIZE = 128
NUM_EPOCHS = 10

normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1=nn.Sequential(
            nn.Conv2d(1,10,5),# 10, 24x24
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2), #10, 12x12
            nn.Conv2d(10,20,3), # 20, 10x10
            nn.ReLU())

        self.fc1 = nn.Sequential(
            nn.Linear(20*10*10,500),
            nn.ReLU(),
            nn.Linear(500,10),
            nn.LogSoftmax(dim=1))

    def forward(self,x):
        in_size = x.size(0)
        out= self.conv1(x) #24
        out = out.view(in_size,-1)#展开成一维，方便进行FC
        out = self.fc1(out)
        return out

model = SimpleNet()

criterion =torch.nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters())

for epoch in range(NUM_EPOCHS):
    for images, labels in tqdm(train_loader):
        optimizer.zero_grad()  # 梯度置零，清空过往梯度
        output = model(images)
        loss = criterion(output, labels)  # 调用内置函数
        loss.backward()  # 反向传播，计算当前梯度
        optimizer.step()  # 根据梯度更新网络参数

    model.eval()
    train_loss = 0
    correct_train = 0
    test_loss = 0
    correct_test = 0
    with torch.no_grad():
        for images, labels in train_loader:
            output = model(images)  # 正向计算预测值
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct_train += pred.eq(labels.view_as(pred)).sum().item()  # 找到正确的预测值
        for images, labels in test_loader:
            output = model(images)  # 正向计算预测值
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct_test += pred.eq(labels.view_as(pred)).sum().item()  # 找到正确的预测值
    train_accuracy=100. * correct_train / len(train_loader.dataset)
    test_accuracy= 100. * correct_test / len(test_loader.dataset)

print('Training accuracy: %0.2f%%' % (train_accuracy))
print('Testing accuracy: %0.2f%%' % (test_accuracy))