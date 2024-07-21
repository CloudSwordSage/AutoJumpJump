# -*- coding: utf-8 -*-
# @Time    : 2024/7/21 19:17
# @Author  : chenfeng
# @Email   : zlf100518@163.com
# @File    : train.py
# @LICENSE : MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from mnist_data import *
from model import MnistNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
batch_size = 128
lr = 0.01
epochs = 300
train_loss = []
test_loss = []
train_acc = []
test_acc = []

train_datas = train_data()
test_datas = test_data()

train_d = DataLoader(train_datas, batch_size=batch_size, shuffle=True)
test_d = DataLoader(test_datas, batch_size=batch_size)

model = MnistNet().to(device)

loss = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=lr)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, train_acc = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        optimizer.zero_grad()
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
    train_loss /= num_batches
    train_acc /= size
    return train_acc, train_loss

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct, test_loss
import time
for epoch in range(epochs):
    s = time.time()
    model.train()
    train_data_acc, train_loss_epoch = train(train_d, model, loss, opt)

    model.eval()
    test_data_acc, test_loss_epoch = test(test_d, model, loss)

    train_loss.append(train_loss_epoch)
    train_acc.append(train_data_acc)
    test_loss.append(test_loss_epoch)
    test_acc.append(test_data_acc)

    template = 'Epoch: {}\tTrain Loss: {:.6f} Acc: {:.4f}% \t Test Loss: {:.6f} Acc: {:.4f}%\tTime: {:.2f}s'
    time_taken = time.time() - s
    print(template.format(epoch + 1, train_loss[-1], train_data_acc * 100, test_loss[-1], test_data_acc * 100, time_taken))

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

epoch_range = range(epochs)

plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)

plt.plot(epoch_range, train_acc, label='Train Acc')
plt.plot(epoch_range, test_acc, label='Test Acc')
plt.legend(loc='lower right')
plt.title('Train and Test Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epoch_range, train_loss, label='Train Loss')
plt.plot(epoch_range, test_loss, label='Test Loss')
plt.legend(loc='upper right')
plt.title('Train and Test Loss')

torch.jit.save(torch.jit.script(model), './model/mnist_cnn.pt')
plt.show()