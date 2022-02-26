import tensorflow.keras as keras
import numpy as np
import math

class DataLoader(keras.utils.Sequence):

    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return math.floor(len(self.labels) / self.batch_size)

    def __getitem__(self, idx):
        x = np.zeros((self.batch_size, 64*64*12*2))
        for i in range(self.batch_size):
            tx = np.load("./features/{}.npz".format(i+(idx*self.batch_size)))['arr_0']
            x[i] = np.unpackbits(tx)

        y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return x, np.array(y)
