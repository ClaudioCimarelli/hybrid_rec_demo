import numpy as np


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


