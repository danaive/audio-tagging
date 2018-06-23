import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold

from utility import *
from data import *
from model import *


def train_on_verified():

    df = pd.read_csv('train.csv')
    verified_samples = df[df.manually_verified == 1]['fname'].values

    with ignore(OSError):
        os.mkdir('checkpoints/verified')
    save_paths = [f'verified/resnet50_r{i:02d}' for i in range(10)]
    round_id = 0

    for ix_tr, ix_val in KFold(n_splits=10).split(verified_samples):
        f_tr, f_val = verified_samples[ix_tr], verified_samples[ix_val]
        with timer('load data'):
            train_loader = DataLoader(DSet(f_tr), batch_size=128, shuffle=True, **kwargs)
            val_loader = DataLoader(DSet(f_val), batch_size=128, **kwargs)

        train(build_resnet50(), train_loader, val_loader, 300, save_paths[round_id])
        round_id += 1

    return save_paths


def make_collision():

    df = pd.read_csv('train.csv')
    unverified = df[df.manually_verified == 0]['fname'].values
    labels = df[df.manually_verified == 0]['label'].values

    with timer('load test data'):
        test_loader = DataLoader(DSet(unverified, labels=False), batch_size=128, **kwargs)
    
    pred = predict(build_resnet50(), test_loader, save_paths)

    df2 = df.set_index('fname')
    for x, y1, y2 in zip(unverified, pred, labels):
        if y1.split()[0] == y2:
            df2.loc[x, 'manually_verified'] = 2
    df2.to_csv('collision.csv')


def train_on_collision():

    df = pd.read_csv('collision.csv')
    train_samples = df[df.manually_verified > 0]['fname'].values

    with ignore(OSError):
        os.mkdir('checkpoints/collision')
    save_paths = [f'collision/resnet50_r{i:02d}' for i in range(10)]
    round_id = 0

    for ix_tr, ix_val in KFold(n_splits=10).split(train_samples):
        f_tr, f_val = train_samples[ix_tr], train_samples[ix_val]
        with timer('load data'):
            train_loader = DataLoader(DSet(f_tr), batch_size=128, shuffle=True, **kwargs)
            val_loader = DataLoader(DSet(f_val), batch_size=128, **kwargs)
        
        train(build_resnet50(), train_loader, val_loader, 300, save_paths[round_id])
        round_id += 1
    
    return save_paths


if __name__ == '__main__':

    save_paths = train_on_collision()
    with timer('load test data'):
        sub = pd.read_csv('sample_submission.csv')
        test_loader = DataLoader(DSet(sub['fname'].values, 'test'), batch_size=128, **kwargs)
    predict(build_resnet50(), test_loader, save_paths, sub)
    