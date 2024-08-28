import numpy as np
import scipy.stats


def compute_wasserstein_correlation(data_one, data_two, n: int, S: int = 40000):
    len_one = data_one.shape[0]
    len_two = data_two.shape[0]

    data_one = np.lib.stride_tricks.sliding_window_view(
        data_one, len_one - n + 1, axis=0
    ).reshape(n, -1)
    data_two = np.lib.stride_tricks.sliding_window_view(
        data_two, len_two - n + 1, axis=0
    ).reshape(n, -1)

    len_one = data_one.shape[0]

    len_one = data_one.shape[1]
    len_two = data_two.shape[1]

    idx_one = np.random.randint(0, len_one, np.minimum(S, len_one))
    idx_two = np.random.randint(0, len_two, np.minimum(S, len_two))

    data_one = data_one[:, idx_one]
    data_two = data_two[:, idx_two]

    if n > 1:
        w = scipy.stats.wasserstein_distance_nd(data_one.T, data_two.T)
    else:
        w = scipy.stats.wasserstein_distance(data_one.flatten(), data_two.flatten())

    return w
