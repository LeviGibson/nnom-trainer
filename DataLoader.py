import tensorflow.keras as keras
import numpy as np
import math
import random

class DataLoader(keras.utils.Sequence):

    def __init__(self, batch_size, name, shuffle=False):
        if not shuffle:
            self.labels = np.load(name + "_labels.npy", mmap_mode='r+')
            self.features = np.load(name + "_features.npy", mmap_mode='r+')
            self.legal = np.load(name + "_legal.npy", mmap_mode='r+')
        else:
            self.labels = np.load(name + "_labels_shuffled.npy", mmap_mode='r+')
            self.features = np.load(name + "_features_shuffled.npy", mmap_mode='r+')
            self.legal = np.load(name + "_legal_shuffled.npy", mmap_mode='r+')

        self.batch_size = batch_size
        self.name = name

    def __len__(self):
        return math.floor(len(self.labels) / self.batch_size)

    def __getitem__(self, idx):
        x = np.zeros((self.batch_size, 64*12))
        y = np.zeros((self.batch_size, 384))
        l = np.zeros((self.batch_size, 384))

        for i in range(self.batch_size):
            index = i+(idx*self.batch_size)
            
            tx = np.zeros((12*64), bool)
            ids = self.features[index]
            for id in ids:
                if id != 0:
                    tx[id] = True

            x[i] = tx
            y[i] = self.labels[index]
            l[i] = self.legal[index]

        return (x, l), y
