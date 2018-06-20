import numpy as np
import pandas as pd
from torch.utils.data import Dataset


N_MFCC = 40
N_STEP = 40
MFCC_TRAIN_PATH = f'mfcc{N_MFCC}_audio_train'
MFCC_TEST_PATH = f'mfcc{N_MFCC}_audio_test'
CLASSES = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"]


class DSet(Dataset):

    def __init__(self, samples, mode='train'):
        self.mode = mode
        self.data = {}
        self.target = np.zeros(len(samples), dtype=np.int)
        if mode == 'train':
            df = pd.read_csv('train.csv').set_index('fname')
        for ix, item in enumerate(samples):
            self.data[ix] = np.load(item).astype(np.float32)
            fname = item[-12:-4] + '.wav'
            if mode == 'train':
                self.target[ix] = CLASSES.index(df.loc[fname, 'label'])
        
    def __len__(self):
        return self.target.shape[0]
    
    def __getitem__(self, ix):
        replay = N_STEP // self.data[ix].shape[1] + 1
        data = np.tile(self.data[ix], replay)
        offset = np.random.randint(data.shape[1] - N_STEP)
        if self.mode == 'train':
            return data[:, offset:offset+N_STEP], self.target[ix]
        else:
            return data[:, offset:offset+N_STEP]
