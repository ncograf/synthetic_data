import warnings
from typing import List, Tuple

import bootstrap
import coarse_fine_volatility
import gain_loss_asymetry
import heavy_tails
import leverage_effect
import linear_unpredictability
import numpy as np
import numpy.typing as npt
import train_garch
import volatility_clustering
from scipy.stats import wasserstein_distance


def boostrap_stylized_facts(
    data: npt.ArrayLike, B: int, S: int, L: int
) -> List[npt.NDArray]:
    data = np.asarray(data)
    assert data.ndim == 2

    def stf_ht(data, qt):
        pd, pb, nd, nb = heavy_tails.discrete_pdf(data)
        (pb, _), _ = heavy_tails.heavy_tails(pd, pb, qt)
        (nb, _), _ = heavy_tails.heavy_tails(nd, nb, qt)

        return np.array([pb, nb]).reshape(-1, 1)

    def stf_cf(data, tau, max_lag):
        _, _, dll, _ = coarse_fine_volatility.coarse_fine_volatility(data, tau, max_lag)
        return dll

    stf_lu = linear_unpredictability.linear_unpredictability
    stf_vc = volatility_clustering.volatility_clustering
    stf_le = leverage_effect.leverage_effect

    stylized_fact_list = [stf_lu, stf_ht, stf_vc, stf_le, stf_cf]
    stylized_fact_arg_list = [
        {"max_lag": 1000},
        {"qt": 0.1},
        {"max_lag": 1000},
        {"max_lag": 60},
        {"max_lag": 50, "tau": 5},
    ]

    out_data = []
    for fact, arg in zip(stylized_fact_list, stylized_fact_arg_list):
        t_data, _ = bootstrap.boostrap_distribution(data, fact, B, S, L, **arg)
        out_data.append(t_data)

    gains, loss = np.nanmean(
        gain_loss_asymetry.gain_loss_asymmetry(
            np.cumsum(data, axis=0), max_lag=80, theta=0.1
        ),
        axis=2,
    )

    out_data.append((gains, loss))

    return out_data


def stylied_facts_from_dist(data: npt.ArrayLike) -> List[npt.NDArray]:
    data = np.asarray(data)
    assert data.ndim == 3

    def stf_ht(data, qt):
        pd, pb, nd, nb = heavy_tails.discrete_pdf(data)
        (pb, _), _ = heavy_tails.heavy_tails(pd, pb, qt)
        (nb, _), _ = heavy_tails.heavy_tails(nd, nb, qt)

        return np.array([pb, nb]).reshape(-1, 1)

    def stf_cf(data, tau, max_lag):
        _, _, dll, _ = coarse_fine_volatility.coarse_fine_volatility(data, tau, max_lag)
        return dll

    stf_lu = linear_unpredictability.linear_unpredictability
    stf_vc = volatility_clustering.volatility_clustering
    stf_le = leverage_effect.leverage_effect

    stylized_fact_list = [stf_lu, stf_ht, stf_vc, stf_le, stf_cf]
    stylized_fact_arg_list = [
        {"max_lag": 1000},
        {"qt": 0.1},
        {"max_lag": 1000},
        {"max_lag": 60},
        {"max_lag": 50, "tau": 5},
    ]

    out_data = []
    for fact, arg in zip(stylized_fact_list, stylized_fact_arg_list):
        t_data = np.asarray([np.mean(fact(d, **arg), axis=1) for d in data]).T
        out_data.append(t_data)

    log_price = np.cumsum(np.concatenate(data, axis=1), axis=0)
    gains, loss = np.nanmean(
        gain_loss_asymetry.gain_loss_asymmetry(log_price, max_lag=80, theta=0.1),
        axis=2,
    )

    out_data.append((gains, loss))

    return out_data


def _stylized_score(
    real_data_stf: List[npt.ArrayLike],
    syn_data_stf: List[npt.ArrayLike],
) -> List[Tuple[npt.ArrayLike, float]]:
    out_data = []

    # last fact is gain loss
    real_data_stf = real_data_stf.copy()
    real_gain, real_loss = real_data_stf.pop()

    syn_data_stf = syn_data_stf.copy()
    syn_gain, syn_loss = syn_data_stf.pop()

    for real, syn in zip(real_data_stf, syn_data_stf):
        n = real.shape[0]
        assert syn.shape[0] == n

        w_dist = []
        for i in range(n):
            std = np.nanstd(real[i, :])
            w_dist.append(wasserstein_distance(real[i, :] / std, syn[i, :] / std))

        out_data.append((np.asarray(w_dist), np.mean(w_dist)))

    #######################
    # wasserstein distance
    #######################

    # comptue stds
    n_gains = np.sum(real_gain)
    mu_gains = np.dot(real_gain, np.arange(1, real_gain.size + 1)) / n_gains
    std_gains = np.sqrt(
        np.dot((np.arange(1, real_gain.size + 1) - mu_gains) ** 2, real_gain) / n_gains
    )

    n_loss = np.sum(real_loss)
    mu_loss = np.dot(real_loss, np.arange(1, real_loss.size + 1)) / n_loss
    std_loss = np.sqrt(
        np.dot((np.arange(1, real_loss.size + 1) - mu_loss) ** 2, real_loss) / n_loss
    )

    values = np.arange(1, real_gain.size + 1)
    values_g = values / std_gains
    values_l = values / std_loss

    gain_score = wasserstein_distance(values_g, values_g, real_gain, syn_gain)
    loss_score = wasserstein_distance(values_l, values_l, real_loss, syn_loss)
    gl_socres = np.asarray([gain_score, loss_score])
    gls = np.mean(gl_socres)
    out_data.append((gl_socres, gls))

    ##################
    # collect results
    ##################

    scores_lists, scores = zip(*out_data)

    return np.mean(scores), scores, scores_lists


def stylized_score(
    real_data: npt.ArrayLike, syn_data: npt.ArrayLike, B: int, S: int, L: int
):
    real_stf = boostrap_stylized_facts(real_data, B, S, L)
    syn_stf = stylied_facts_from_dist(syn_data)

    return _stylized_score(real_stf, syn_stf)


if __name__ == "__main__":
    import load_data

    sp500 = load_data.load_log_returns("sp500")
    smi = load_data.load_log_returns("smi")

    B = 5
    S = 24
    L = 4096

    # comptue samples for garch
    with warnings.catch_warnings(action="ignore"):
        garch_b = []
        for i in range(B):
            garch_sample = train_garch.sample_garch(
                "/home/nico/thesis/code/data/cache/garch_experiments/GARCH_ged_2024_07_14-02_42_08",
                n_stocks=S,
                len=L,
            )
            print(f"sample garch {i}")
            garch_b.append(garch_sample)

    syn_data = np.asarray(garch_b)

    mu_score, scores, _ = stylized_score(sp500, syn_data, B, S, L)
    print(scores, mu_score)
