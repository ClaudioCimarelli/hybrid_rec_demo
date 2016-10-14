from util import *


def user_update(u_i, v, bias, profile, epochs=150, alpha0=0.023, beta=0.05):
    profile = np.reshape(profile, (1, -1))
    u_i = np.reshape(u_i, (1, -1))
    nz_profile = non_zero_matrix(profile)
    vel_u = np.zeros_like(u_i)
    u_prev = np.zeros_like(u_i)
    u_prev[...] = u_i

    f = (np.dot(u_i, v.T) + bias) * nz_profile
    err = profile - f
    for epoch in range(epochs):

        alpha = max(alpha0 / (1 + (epoch / 150)), 0.01)
        mu = min(0.89, 1.2 / (1 + np.exp(-epoch / 100)))

        u_ahead = u_i + (mu * vel_u)

        delta__u = np.dot(2 * alpha * err, alpha * v) - (2 * alpha * beta * u_ahead)

        vel_u *= mu
        vel_u += delta__u
        u_i += vel_u

        f = (np.dot(u_i, v.T) + bias) * nz_profile
        err = profile - f

    return u_i

def new_user_update(v, bias, profile, items_rated):
    u_b = np.random.uniform(-0.05, 0.05, len(v[0]))
    u_b[...] = user_update(u_b, v[items_rated, :], bias, profile)
    imf_pred = np.dot(u_b, v.T) + bias
    imf_pred = np.maximum(np.minimum(imf_pred, 5), 1)
    return imf_pred


def update_model(u_batch, u_i, i):
    np.save('data/u_online', u_batch)
