import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold

from utility import *
from data import *
from model import *


if __name__ == '__main__':

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
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
