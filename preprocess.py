import os
import multiprocessing
from collections import defaultdict
from glob import glob

import librosa
import numpy as np

from utility import *


SAMPLE_RATE = 44100
PARALLEL_CORE = multiprocessing.cpu_count() - 1
N_MFCC = 40


def work(audios):

    print(len(audios))
    print(int(audios[0][-12:-4], 16) % PARALLEL_CORE, PARALLEL_CORE)
    for audio in audios:
        wav, _ = librosa.core.load(audio, sr=SAMPLE_RATE)
        try:
            mfcc = librosa.feature.mfcc(wav, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
            np.save(f'mfcc{N_MFCC}_{audio[:-4]}.npy', mfcc)
        except:
            print('bad file', audio)


def grouping(path):

    audios = glob(f'{path}/*.wav')
    groups = defaultdict(list)
    for audio in audios:
        group_id = int(audio[-12:-4], 16) % PARALLEL_CORE
        groups[group_id].append(audio)
    return groups


def find_shortest(path):

    audios = glob(f'{path}/*.npy')
    size = (50, 0)
    for audio in audios:
        mfcc = np.load(audio)
        size = min(size, mfcc.shape)
    print(size)


if __name__ == '__main__':

    with ignore(OSError):
        os.mkdir(f'mfcc{N_MFCC}_audio_train')
    with ignore(OSError):
        os.mkdir(f'mfcc{N_MFCC}_audio_test')

    # train_groups = grouping('audio_train')
    # with timer('transforming train data'):
    #     train_pool = multiprocessing.Pool(processes=PARALLEL_CORE)
    #     for group in train_groups:
    #         train_pool.apply_async(work, (train_groups[group],))
    #     train_pool.close()
    #     train_pool.join()

    test_groups = grouping('audio_test')
    with timer('transforming test data'):
        # test_pool = multiprocessing.Pool(processes=PARALLEL_CORE)
        for group in test_groups:
            # test_pool.apply_async(work, (test_groups[group],))
            work(test_groups[group])
        # test_pool.close()
        # test_pool.join()
    
    # find_shortest('mfcc_audio_test')