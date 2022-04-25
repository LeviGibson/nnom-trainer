from npy_append_array import NpyAppendArray
import numpy as np
import random

def shuffle(fname):
    infile1 = np.load(fname + "_features.npy", mmap_mode='r')
    infile2 = np.load(fname + "_labels.npy", mmap_mode='r')
    infile3 = np.load(fname + "_legal.npy", mmap_mode='r')
    features1 = NpyAppendArray(fname + "_features_shuffled.npy")
    features2 = NpyAppendArray(fname + "_labels_shuffled.npy")
    features3 = NpyAppendArray(fname + "_legal_shuffled.npy")

    trans = list(range(infile1.shape[0]))
    random.shuffle(trans)

    for tid, t in enumerate(trans):
        features1.append(np.array([infile1[t]]))
        features2.append(np.array([infile2[t]]))
        features3.append(np.array([infile3[t]]))
        print(tid / len(trans))

shuffle("train")