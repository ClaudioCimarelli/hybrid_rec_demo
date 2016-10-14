import numpy as np
from scipy.sparse import *
import os


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


def expert_base(batch_matrix, max_users=4000):
    nz_batch = non_zero_matrix(batch_matrix)
    ratings_count = np.sum(nz_batch, axis=1)
    sort_by_ratings_index = np.argsort(ratings_count, kind='mergesort')
    experts_index = np.sort(sort_by_ratings_index[-max_users:])
    private_users_index = np.sort(sort_by_ratings_index[:-max_users])
    return experts_index, private_users_index

def non_zero_matrix(r):
    with np.errstate(divide='ignore', invalid='ignore'):
        nz_r = np.nan_to_num(np.divide(r, r))
    return nz_r


def calc_rmse(real_values, prediction):
    mask = non_zero_matrix(real_values)
    error = (real_values - prediction) * mask
    error **= 2
    RMSE = (np.sum(error) / np.sum(mask)) ** (1 / 2)
    return RMSE


