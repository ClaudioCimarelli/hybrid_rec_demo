import numpy as np
from collections import OrderedDict
from user_based_predictions import user_based_pred
from online_updates import new_user_update
from hybrid_rec import hybrid_rec


def user_rec(n_user, cluster, v, bias):
    recommenders_results = OrderedDict([])

    items_rated = np.nonzero(cluster[n_user, :])[0]

    ub_pred = user_based_pred(cluster)[n_user, :]
    ub_items = np.argsort(ub_pred)[::-1]
    ub_items = np.delete(ub_items, items_rated)
    ub_ratings = ub_pred[ub_items]
    ub_rec = dict(items=ub_items[:25], ratings=ub_ratings[:25])

    imf_pred = new_user_update(v, bias, cluster[n_user, items_rated], items_rated)
    imf_items = np.argsort(imf_pred)[::-1]
    imf_items = np.delete(imf_items, items_rated)
    imf_ratings = imf_pred[imf_items]
    imf_rec = dict(items=imf_items[:25], ratings=imf_ratings[:25])

    hyb_pred = hybrid_rec(imf_pred, ub_pred, alpha=0.8)
    hyb_items = np.argsort(hyb_pred)[::-1]
    hyb_items = np.delete(hyb_items, items_rated)
    hyb_ratings = hyb_pred[hyb_items]
    hyb_rank_imf = relative_rank(imf_items, hyb_items) + 1
    hyb_rank_ub = relative_rank(ub_items, hyb_items) + 1
    hyb_rec = dict(items=hyb_items[:25], ratings=hyb_ratings[:25], rank_imf=hyb_rank_imf, rank_ub=hyb_rank_ub)

    recommenders_results.update({'Hybrid recommender': hyb_rec})
    recommenders_results.update({'Incremental Matrix Factorization' : imf_rec})
    recommenders_results.update({'User Based Collaborative Filtering' : ub_rec})

    return recommenders_results


def relative_rank(master, search):
    sorti = np.argsort(master)
    sorti_inv = np.empty(sorti.shape, dtype=np.int64)
    sorti_inv[sorti] = np.arange(sorti.size)
    sorti_s = np.argsort(search)
    final_inds = sorti_s[sorti_inv]

    # # sorting permutation and its reverse
    # sorti = np.argsort(master)
    # sorti_inv = np.zeros_like(sorti)
    # sorti_inv[sorti] = np.arange(sorti.size)
    #
    # # get indices in sorted version
    # tmpind = np.searchsorted(master, search, sorter=sorti)
    #
    # # transform indices back to original array with inverse permutation
    # final_inds = tmpind[sorti_inv]

    return final_inds
