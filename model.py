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
        fc_indim = (input_shape // 16) * (n_mfcc // 16) * 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 9), padding=(1, 4))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 9), padding=(1, 4))
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 9), padding=(1, 4))
        self.bn4 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(fc_indim, 64)
        self.bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, len(CLASSES))

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.bn3(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.bn4(self.conv4(x)), 2))
        x = F.relu(self.bn(self.fc1(x.view(x.size(0), -1))))
        return F.log_softmax(self.fc2(x), dim=1)


def mapk(output, target, k):
    
    def apk(pred, actual):
        for i, p in enumerate(pred):
            if p == actual and p not in pred[:i]:
                return 1 / (i + 1)
        return 0
    
    _, pred = torch.sort(-output, dim=1)
    return np.mean([apk(p, t) for p, t in zip(pred[:, :k], target)])


def train(model, n_epoch=20, save_path=None):

    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.RMSprop(model.parameters())
    best_score = 0.0

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

        if epoch % 5:
            continue
        model.eval()
        pred, actual = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss.append(loss.item())
                output, target = output.cpu().detach(), target.cpu().detach()
                pred.append(output)
                actual.append(target)
            actual, pred = torch.cat(actual, dim=0), torch.cat(pred, dim=0)
            map1, map3 = mapk(pred, actual, 1), mapk(pred, actual, 3)
        if map3 > best_score:
            checkpoint = {
                'state': model.state_dict(),
                'epoch': epoch,
                'map3': map3,
            }
            print('updating checkpoint')
            torch.save(checkpoint, f'checkpoints/{save_path}.pth')
            best_score = map3
        print(f'val loss: {np.mean(val_loss):.4f}\tmap@1: {np.mean(map1):.4f}\tmap@3: {np.mean(map3):.4f}')
        print('=' * 50)
    print(f'best validation map@3 {checkpoint["map3"]} at epoch {checkpoint["epoch"]}')


def predict(checkpoints, sub):

    def predict_once(cp):
        model = CNN(N_STEP)
        checkpoint = torch.load(f'checkpoints/{cp}.pth')
        print(f'using model: map@3 {checkpoint["map3"]:.4f} at epoch {checkpoint["epoch"]}')
        model.load_state_dict(checkpoint['state'])
        model.eval()
        pred = []
        with torch.no_grad():
            for data in test_loader:
                output = model(data)
                pred.append(output)
        return torch.cat(pred, dim=0)

    result = [predict_once(cp) for cp in checkpoints]
    result = torch.cat(result, dim=0).view(len(checkpoints), -1, len(CLASSES)).mean(dim=0)
    _, result = torch.sort(-result, dim=1)
    labels = []
    for pred in result:
        labels.append(' '.join(map(lambda x: CLASSES[x], pred[:3])))
    sub['label'] = labels
    sub.to_csv('submission/last_prediction.csv', index=False)


if __name__ == '__main__':

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_samples = pd.read_csv('train.csv')['fname'].values
    f_tr, f_val = train_test_split(train_samples, test_size=0.1)
    with timer('load data'):
        train_loader = DataLoader(DSet(f_tr), batch_size=128, shuffle=True, **kwargs)
        val_loader = DataLoader(DSet(f_val), batch_size=128, **kwargs)

    train(CNN(N_STEP), 200, 'firstattempt')
    with timer('load test data'):
        sub = pd.read_csv('sample_submission.csv')
        test_loader = DataLoader(DSet(sub['fname'].values, 'test'), batch_size=128, **kwargs)
    predict(['firstattempt'], sub)
    