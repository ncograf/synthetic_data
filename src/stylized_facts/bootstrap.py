import coarse_fine_volatility
import leverage_effect
import linear_unpredictability
import load_data
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import volatility_clustering
from scipy.stats import wasserstein_distance


def smooth(data):
    n_smooth = 2
    cs_data = np.cumsum(data, axis=0)
    data = np.copy(data)  # copy array

    # handle edge cases
    # for i in range(1,n_smooth):
    #    data[i] = (cs_data[2 * i]) / (2 * i + 1)
    #    data[-(i+1)] = (cs_data[-1] - cs_data[-2 * (i+1)])  / (2 * i + 1)
    data[n_smooth + 1 : -n_smooth] = (
        cs_data[2 * n_smooth + 1 :] - cs_data[: -(2 * n_smooth + 1)]
    ) / (2 * n_smooth + 1)
    return data


def boostrap_distribution(
    data: npt.ArrayLike, st_fact, B: int, S: int, L: int, **kwargs
):
    # estimator
    stf = np.mean(np.asarray(st_fact(data, **kwargs)), axis=1)
    stf = stf.reshape(-1, 1)

    # boostrap
    bstrap = []
    for _ in range(B):
        data_b = []
        for _ in range(S):
            ind = np.random.randint(0, data.shape[1])
            # start = np.random.randint(0, data.shape[0] - L)
            start = 0
            data_b.append(data[start : start + L, ind])

        data_b = np.asarray(data_b).T
        bstrap.append(np.mean(np.asarray(st_fact(data_b, **kwargs)), axis=1))

    bstrap = np.asarray(bstrap).T
    bstrap = np.sort(bstrap, axis=1)

    # compute efrons boostrap distribution
    b_dist = 2 * stf - bstrap

    return b_dist, stf


def plot_dist(bstrap, theta_hat, plot, label):
    sixty = np.array([B * 0.2, B * 0.8]).astype(int)
    ninty_five = np.array([B * 0.025, B * 0.975]).astype(int)
    sixty = bstrap[:, sixty]
    ninty_five = bstrap[:, ninty_five]

    x = np.arange(1, theta_hat.shape[0] + 1)

    plot.fill_between(x, sixty[:, 0], sixty[:, 1], fc="gray", alpha=0.2)
    plot.fill_between(x, ninty_five[:, 0], ninty_five[:, 1], fc="gray", alpha=0.2)
    plot.plot(x, sixty, linewidth=1, c="gray", alpha=0.3)
    plot.plot(x, ninty_five, linewidth=1, c="gray", alpha=0.3)
    plot.plot(x, theta_hat, linewidth=3, label=label)

    return plot


def plot_stf(st_fact, **kwargs):
    stf_b_sp500, stf_sp500 = boostrap_distribution(sp500, st_fact, B, S, L, **kwargs)
    stf_b_smi, stf_smi = boostrap_distribution(smi, st_fact, B, S, L, **kwargs)
    stf_b_dax, stf_dax = boostrap_distribution(dax, st_fact, B, S, L, **kwargs)

    sp_smi = []
    sp_dax = []
    dax_smi = []
    for i in range(stf_b_dax.shape[0]):
        dax_smi.append(wasserstein_distance(stf_b_dax[i, :], stf_b_smi[i, :]))
        sp_dax.append(wasserstein_distance(stf_b_dax[i, :], stf_b_sp500[i, :]))
        sp_smi.append(wasserstein_distance(stf_b_smi[i, :], stf_b_sp500[i, :]))

    plt.plot(dax_smi, label=f"dax smi {np.mean(dax_smi)}", linestyle="none", marker="o")
    plt.plot(sp_dax, label=f"sp500 dax {np.mean(sp_dax)}", linestyle="none", marker="o")
    plt.plot(sp_smi, label=f"sp500 smi {np.mean(sp_smi)}", linestyle="none", marker="o")
    plt.legend()
    plt.show()

    plot = plt.gca()

    plot_dist(stf_b_smi, stf_smi.flatten(), plot, label="smi")
    plot_dist(stf_b_dax, stf_dax.flatten(), plot, label="dax")
    plot_dist(stf_b_sp500, stf_sp500.flatten(), plot, label="sp")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    sp500 = load_data.load_log_returns("sp500", min_len=9216)
    smi = load_data.load_log_returns("smi", min_len=4096)
    dax = load_data.load_log_returns("dax", min_len=4096)

    B = 500
    S = 24
    L = 4096

    le = leverage_effect.leverage_effect_torch
    le_args = {"max_lag": 100}
    # plot_stf(le, **le_args)

    lu = linear_unpredictability.linear_unpredictability
    lu_args = {"max_lag": 100}
    # plot_stf(lu, **lu_args)

    vc = volatility_clustering.volatility_clustering_torch
    vc_args = {"max_lag": 100}
    # plot_stf(vc, **vc_args)

    def cf(data, tau, max_lag):
        _, _, dll, _ = coarse_fine_volatility.coarse_fine_volatility(data, tau, max_lag)
        return dll

    cf_args = {"max_lag": 150, "tau": 5}
    # plot_stf(cf, **cf_args)
