import numpy as np
from cosinesim import cos_sim


def user_data(n_user, cluster, experts_matrix, experts_u, user_u):
    data = {}
    pass
    data['user_ratings'] = ratings_by_user(cluster[n_user, :])
    items_rated = np.nonzero(cluster[n_user, :])[0]
    neighbourhood = cluster_neighbours_items(n_user, cluster)
    data['cluster_neighbours'] = top_items_neighbours(neighbourhood, cluster, items_rated)['items']
    neighbourhood = experts_neighbours_items(user_u, experts_u)
    data['experts_neighbours'] = top_items_neighbours(neighbourhood, experts_matrix, items_rated)['items']
    return data


def cluster_neighbours_items(n_user, cluster, n=10):
    sim = cos_sim(cluster)[n_user, :]
    best_neighbour = np.argsort(sim)[::-1]
    best_neighbour = np.delete(best_neighbour, np.where(best_neighbour == n_user)[0])[:n]
    return best_neighbour


def experts_neighbours_items(user_u, experts_u, n=10):
    sim_set = np.append(experts_u, [user_u], axis=0)
    sim = cos_sim(sim_set)[-1, :-1]
    best_neighbour = np.argsort(sim)[::-1][:n]
    return best_neighbour


def ratings_by_user(neighbour_ratings, user_items=None):
    neighbour_items = np.nonzero(neighbour_ratings)[0]
    # if not(user_items is None):
    #     neighbour_items = np.setdiff1d(neighbour_items, user_items)
    items = np.argsort(neighbour_ratings)[::-1]
    items_inv = np.zeros_like(items)
    items_inv[items] = np.arange(items.size)
    mask = np.ones_like(neighbour_ratings,dtype=bool)
    mask[neighbour_items] = False
    items = np.delete(items, items_inv[mask])
    ratings = neighbour_ratings[items]
    user_ratings = {'items': items, 'ratings': ratings}
    return user_ratings


def top_items_neighbours(neighbours_list, neighbours_ratings, items_rated, n=3):
    neighbours_items = {'neighbours': neighbours_list,
                        'items': np.empty((neighbours_list.size, n), dtype='int16')}

    for i, neighbour in enumerate(neighbours_list):
        neigh_items = ratings_by_user(neighbours_ratings[neighbour, :], items_rated)['items']
        neighbours_items['items'][i] = neigh_items[:n]
    return neighbours_items
