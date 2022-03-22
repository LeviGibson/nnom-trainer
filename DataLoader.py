import tensorflow.keras as keras
import numpy as np
import math
import random

class DataLoader(keras.utils.Sequence):

    def __init__(self, batch_size, name):
        self.labels = np.load(name + "_labels.npy")
        self.batch_size = batch_size
        self.index_transformation = list(range(len(self.labels)))
        self.randomise()
        self.name = name

    def randomise(self):
        for i in range(len(self.index_transformation)):
            target = random.randint(0, len(self.index_transformation)-1)
            self.index_transformation[target], self.index_transformation[i] = self.index_transformation[i], self.index_transformation[target]

    def __len__(self):
        return math.floor(len(self.labels) / self.batch_size)

    def __getitem__(self, idx):
        x = np.zeros((self.batch_size, 64*64*12*2))
        y = np.zeros((self.batch_size, 384))
        for i in range(self.batch_size):
            index = self.index_transformation[i+(idx*self.batch_size)]
            
            tx = np.load(self.name + "_features/{}.npz".format(index))['arr_0']
            x[i] = np.unpackbits(tx)
            y[i] = self.labels[index]

        return x, y
