import sys
import util
sys.path.append('..')

import numpy as np
import os
from time import time
from collections import Counter
import random
from matplotlib import pyplot as plt


train_data_size = 15000
validation_data_start = 13000

data_dir = '/home/sparsh/data'
def mnist():
    
    tempX, tempY = np.load("data_file.npy"), util.OneHot(np.load("labels_file.npy"))

    tempX = tempX.reshape((784, tempX.shape[2]))
    tempX = tempX.transpose()


    trX = tempX[:train_data_size]

    trY = tempY[:train_data_size]

    teX = tempX[train_data_size:]

    teY = tempY[train_data_size:]

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    np.save("formatted_data", trX)
    np.save("formatted_labels", trY)
    np.save("formatted_testdata", teX)
    np.save("formatted_testlabels",teY)

    return trX, teX, trY, teY

def mnist_with_valid_set():
    trX, teX, trY, teY = mnist()

    train_inds = np.arange(len(trX))
    np.random.shuffle(train_inds)
    trX = trX[train_inds]
    trY = trY[train_inds]
    #trX, trY = shuffle(trX, trY)
    vaX = trX[validation_data_start:]
    vaY = trY[validation_data_start:]
    trX = trX[:validation_data_start]
    trY = trY[:validation_data_start]

    return trX, vaX, teX, trY, vaY, teY
