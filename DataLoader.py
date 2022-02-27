import tensorflow.keras as keras
import numpy as np
import math
import random

class DataLoader(keras.utils.Sequence):

    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.index_transformation = range(len(labels))

    def randomise(self):
        for i in range(len(self.index_transformation)):
            target = random.randint(0, len(self.index_transformation)-1)

            tmp = self.index_transformation[target]
            self.index_transformation[target] = self.index_transformation[i]
            self.index_transformation[i] = tmp

            tmp = self.labels[i]
            self.labels[i] = self.labels[target]
            self.labels[target] = tmp

    def __len__(self):
        return math.floor(len(self.labels) / self.batch_size)

    def __getitem__(self, idx):
        x = np.zeros((self.batch_size, 64*64*12*2))
        for i in range(self.batch_size):
            index = self.index_transformation[i+(idx*self.batch_size)]
            
            tx = np.load("./features/{}.npz".format(index))['arr_0']
            x[i] = np.unpackbits(tx)

        y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return x, np.array(y)
