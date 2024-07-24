import warnings
from typing import Callable, Tuple

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


def _stf_score(
    stf: Callable,
    real_data: npt.ArrayLike,
    syn_data: npt.ArrayLike,
    B: int,
    S: int,
    L: int,
    **kwargs,
) -> Tuple[npt.ArrayLike, float]:
    real_data = np.asarray(real_data)
    syn_data = np.asarray(syn_data)

    # two dimensional data needs to be boostrapped
    if real_data.ndim == 2:
        real_data, _ = bootstrap.boostrap_distribution(
            real_data, stf, B, S, L, **kwargs
        )
    else:
        assert real_data.ndim == 3
        real_data = np.asarray([np.mean(stf(d, **kwargs), axis=1) for d in real_data]).T

    if syn_data.ndim == 2:
        dist_b, _ = bootstrap.boostrap_distribution(syn_data, stf, B, S, L, **kwargs)
    else:
        assert syn_data.ndim == 3
        dist_b = np.asarray([np.mean(stf(d, **kwargs), axis=1) for d in syn_data]).T

    n = real_data.shape[0]
    assert dist_b.shape[0] == real_data.shape[0]

    w_dist = []
    for i in range(n):
        std = np.std(real_data[i, :])
        w_dist.append(wasserstein_distance(real_data[i, :] / std, dist_b[i, :] / std))

    return np.asarray(w_dist), np.mean(w_dist)


def stylized_score(
    real_data: npt.ArrayLike, syn_data: npt.ArrayLike, B: int, S: int, L: int
):
    tail_quantile = 0.1
    tau = 5
    theta = 0.1

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

    lu_scores, lus = _stf_score(stf_lu, real_data, syn_data, B, S, L, max_lag=1000)
    ht_scores, hts = _stf_score(stf_ht, real_data, syn_data, B, S, L, qt=tail_quantile)
    vc_scores, vcs = _stf_score(stf_vc, real_data, syn_data, B, S, L, max_lag=1000)
    le_scores, les = _stf_score(stf_le, real_data, syn_data, B, S, L, max_lag=100)
    cf_scores, cfs = _stf_score(
        stf_cf, real_data, syn_data, B, S, L, max_lag=40, tau=tau
    )

    gl_max = 100
    values = np.arange(1, gl_max + 2, 1)

    if real_data.ndim == 3:
        real_data = np.concatenate(real_data.tolist(), axis=1)
    if syn_data.ndim == 3:
        syn_data = np.concatenate(syn_data.tolist(), axis=1)

    real_gains, real_loss = np.nanmean(
        gain_loss_asymetry.gain_loss_asymmetry(real_data, max_lag=gl_max, theta=theta),
        axis=2,
    )
    syn_gains, syn_loss = np.nanmean(
        gain_loss_asymetry.gain_loss_asymmetry(syn_data, max_lag=gl_max, theta=theta),
        axis=2,
    )

    n_gains = np.sum(real_gains)
    mu_gains = np.dot(real_gains, np.arange(1, real_gains.size + 1)) / n_gains
    std_gains = np.sqrt(
        np.dot((np.arange(1, real_gains.size + 1) - mu_gains) ** 2, real_gains)
        / n_gains
    )

    n_loss = np.sum(real_loss)
    mu_loss = np.dot(real_loss, np.arange(1, real_loss.size + 1)) / n_loss
    std_loss = np.sqrt(
        np.dot((np.arange(1, real_loss.size + 1) - mu_loss) ** 2, real_loss) / n_loss
    )

    values_g = values / std_gains
    values_l = values / std_loss
    gain_score = wasserstein_distance(values_g, values_g, real_gains, syn_gains)
    loss_score = wasserstein_distance(values_l, values_l, real_loss, syn_loss)
    gl_socres = np.asarray([gain_score, loss_score])
    gls = np.mean(gl_socres)

    scores = np.array((lus, hts, vcs, les, cfs, gls))
    score_arrays = [lu_scores, ht_scores, vc_scores, le_scores, cf_scores]

    return scores, np.mean(scores), score_arrays


if __name__ == "__main__":
    import load_data

    sp500 = load_data.load_log_returns("sp500")
    smi = load_data.load_log_returns("smi")

    B = 50
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

    syn_data = np.asarray(garch_b[1])

    scores, mu_score, _ = stylized_score(sp500, syn_data, B, S, L)
    print(scores, mu_score)
