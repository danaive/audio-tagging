import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utility import *
from data import *


assert torch.__version__ >= '0.4.0'
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class CNN(nn.Module):

    def __init__(self, input_shape, n_mfcc=N_MFCC):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 9), padding=(1, 4))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 9), padding=(1, 4))
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 9), padding=(1, 4))
        self.bn4 = nn.BatchNorm2d(32)
        fc_indim = (input_shape // 8) * (n_mfcc // 8) * 32
        self.fc = nn.Linear(fc_indim, len(CLASSES))

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.bn3(self.conv3(x)), 2))
        # x = F.relu(F.max_pool2d(self.bn4(self.conv4(x))))
        return F.log_softmax(self.fc(x.view(x.size(0), -1)), dim=1)


def mapk(output, target, k):
    
    def apk(pred, actual):
        for i, p in enumerate(pred):
            if p == actual and p not in pred[:i]:
                return 1 / (i + 1)
        return 0
    
    _, pred = torch.sort(-output, dim=1)
    return np.mean([apk(p, t) for p, t in zip(pred[:, :k], target)])


def train(model, n_epoch=20, eval_repeat=10):

    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.RMSprop(model.parameters())
    assert len(train_loader)

    for epoch in range(n_epoch):
        epoch += 1
        print(f'Train Epoch: {epoch}')
        tr_loss, val_loss = [], []

        model.train()
        map1, map3 = [], []
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            tr_loss.append(loss.item())
            output, target = output.cpu().detach(), target.cpu().detach()
            map1.append(mapk(output, target, 1))
            map3.append(mapk(output, target, 3))
        print(f'train loss: {np.mean(tr_loss):.4f}\tmap@1: {np.mean(map1):.4f}\tmap@3: {np.mean(map3):.4f}')

        if epoch % 10:
            continue
        model.eval()
        map1, map3 = [], []
        pred, actual = [], []
        with torch.no_grad():
            for er in range(eval_repeat):
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss.append(loss.item())
                    output, target = output.cpu().detach(), target.cpu().detach()
                    pred.append(output)
                    actual.append(target)
            actual = torch.cat(actual, dim=0).view(10, -1)[0]
            pred = torch.cat(pred, dim=0).view(10, -1, 41).mean(dim=0)
            map1 = mapk(pred, actual, 1)
            map3 = mapk(pred, actual, 3)
        print(f'val loss: {np.mean(val_loss):.4f}\tmap@1: {np.mean(map1):.4f}\tmap@3: {np.mean(map3):.4f}')
        print('=' * 20)


if __name__ == '__main__':

    samples = glob.glob(f'mfcc{N_MFCC}_audio_train/*.npy')
    f_tr, f_val = train_test_split(samples, test_size=0.3)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    with timer('load data'):
        train_loader = DataLoader(DSet(f_tr), batch_size=128, shuffle=True, **kwargs)
        val_loader = DataLoader(DSet(f_val), batch_size=128, **kwargs)

    train(CNN(40), 200)