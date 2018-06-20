import os
import multiprocessing
from collections import defaultdict
from glob import glob

import librosa
import numpy as np

from utility import *


SAMPLE_RATE = 44100
PARALLEL_CORE = multiprocessing.cpu_count() - 1


def work(audios):

    for audio in audios:
        wav, _ = librosa.core.load(audio, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(wav, sr=SAMPLE_RATE, n_mfcc=40)
        np.save(f'mfcc_{audio[:-4]}.npy', mfcc)


def grouping(path):

    audios = glob(f'{path}/*.wav')
    groups = defaultdict(list)
    for audio in audios:
        group_id = int(audio[12:-4], 16) % PARALLEL_CORE
        groups[group_id].append(audio)
    return groups


if __name__ == '__main__':

    with ignore(OSError):
        os.mkdir('mfcc_audio_train')
    with ignore(OSError):
        os.mkdir('mfcc_audio_test')

    with timer('grouping train data'):
        train_groups = grouping('audio_train')
    with timer('transforming train data'):
        train_pool = multiprocessing.Pool(processes=PARALLEL_CORE)
        for group in train_groups:
            train_pool.apply_async(work, (train_groups[group],))
        train_pool.close()
        train_pool.join()

    with timer('grouping test data'):
        test_groups = grouping('audio_test')
    with timer('transforming test data'):
        test_pool = multiprocessing.Pool(processes=PARALLEL_CORE)
        for group in test_groups:
            test_pool.apply_async(work, (test_groups[group],))
        test_pool.close()
        test_pool.join()
