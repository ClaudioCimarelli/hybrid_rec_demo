import numpy as np
from cosinesim import cos_sim


def user_data(n_user, cluster, experts_u, user_u):
    data = {}
    pass
    data['user_ratings'] = ratings_by_user(n_user, cluster)
    data['cluster_neighbours'] = cluster_neighbours_items(n_user, cluster)
    data['experts_neighbours'] = experts_neighbours_items(user_u, experts_u)
    return data





def cluster_neighbours_items(n_user, cluster, n=10):
    sim = cos_sim(cluster)[n_user, :]
    best_neighbour = np.argsort(sim)[::-1]
    best_neighbour = np.delete(best_neighbour, np.where(best_neighbour == n_user)[0])[:n]
    return top_items_neighbours(best_neighbour, cluster)


def experts_neighbours_items(user_u, experts_u, n=10):
    sim_set = np.append(experts_u, [user_u], axis=0)
    sim = cos_sim(sim_set)[-1, :-1]
    best_neighbour = np.argsort(sim)[::-1][:n]
    return top_items_neighbours(best_neighbour, experts_u)


def ratings_by_user(n_user, cluster):
    items_rated = np.nonzero(cluster[n_user, :])[0]
    ratings = cluster[n_user, items_rated]
    items_rated = np.argsort(ratings)[::-1]
    ratings[...] = ratings[items_rated]
    user_ratings = {'items': items_rated, 'ratings': ratings}
    return user_ratings


def top_items_neighbours(neighbours_list, cluster, n=3):
    neighbours_items = {'neighbour': neighbours_list,
                        'items': np.empty((neighbours_list.size, n), dtype='int16')}

    for i, neighbour in enumerate(neighbours_list):
        neigh_items = ratings_by_user(neighbour, cluster)['items']
        neighbours_items['items'][i] = neigh_items[:n]
    return neighbours_items
