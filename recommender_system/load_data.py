from scipy.sparse import *
import os
import numpy as np


def load_data():
    path = os.path.dirname(__file__)
    try:
        ratings_dataset = np.load(path + '/ml-1m/ratings.npy')
    except:
        ratings_dataset = np.loadtxt(path + "ml-1m/ratings.dat", dtype=np.int32, delimiter='::', usecols=(0, 1, 2))
        np.save('/data/ratings', ratings_dataset)

    row = ratings_dataset[:, 0] - 1
    col = ratings_dataset[:, 1] - 1
    data = ratings_dataset[:, 2]
    batch_matrix = coo_matrix((data, (row, col))).toarray()
    return batch_matrix