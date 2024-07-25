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
import train_fingan
import train_garch
import volatility_clustering
from scipy.stats import wasserstein_distance


def _stf_ht(data, tq):
    pd, pb, nd, nb = heavy_tails.discrete_pdf(data)
    (pb, pa), _ = heavy_tails.heavy_tails(pd, pb, tq)
    (nb, na), _ = heavy_tails.heavy_tails(nd, nb, tq)

    return np.array([pb, nb, pa, na]).reshape(-1, 1)


def _stf_cf(data, tau, max_lag):
    _, _, dll, _ = coarse_fine_volatility.coarse_fine_volatility(data, tau, max_lag)
    return dll


_stf_lu = linear_unpredictability.linear_unpredictability
_stf_vc = volatility_clustering.volatility_clustering
_stf_le = leverage_effect.leverage_effect

_stylized_fact_list = [_stf_lu, _stf_ht, _stf_vc, _stf_le, _stf_cf]
_stylized_fact_arg_list = [
    {"max_lag": 1000},
    {"tq": 0.1},
    {"max_lag": 1000},
    {"max_lag": 100},
    {"max_lag": 120, "tau": 5},
]
_glarg = {"max_lag": 1000, "theta": 0.1}


def compute_mean_stylized_fact(data: npt.ArrayLike):
    data = np.asarray(data)
    assert data.ndim == 2

    out_data = []

    # lin unpred
    t_data = np.mean(
        linear_unpredictability.linear_unpredictability(
            data, **(_stylized_fact_arg_list[0])
        ),
        axis=1,
    )
    out_data.append(t_data)

    # heavy tails
    pd, pbin, nd, nbin = heavy_tails.discrete_pdf(data)
    (pb, pa), pos_x = heavy_tails.heavy_tails(pd, pbin, **_stylized_fact_arg_list[1])
    (nb, na), neg_x = heavy_tails.heavy_tails(nd, nbin, **_stylized_fact_arg_list[1])
    t_data = (np.array([pb, nb, pa, na]), pos_x, neg_x, pd, pbin, nd, nbin)
    out_data.append(t_data)

    # vol clustering
    t_data = np.mean(
        volatility_clustering.volatility_clustering(
            data, **(_stylized_fact_arg_list[2])
        ),
        axis=1,
    )
    out_data.append(t_data)

    # leverage effect
    t_data = np.mean(
        leverage_effect.leverage_effect(data, **(_stylized_fact_arg_list[3])), axis=1
    )
    out_data.append(t_data)

    # coarse fine volatility
    ll_mean, ll_x, dll_mean, dll_x = coarse_fine_volatility.coarse_fine_volatility(
        data, **(_stylized_fact_arg_list[4])
    )
    ll_mean, dll_mean = np.mean(ll_mean, axis=1), np.mean(dll_mean, axis=1)
    out_data.append((ll_mean, ll_x, dll_mean, dll_x))

    # gain loss asym
    gains, loss = np.nanmean(
        gain_loss_asymetry.gain_loss_asymmetry(np.cumsum(data, axis=0), **_glarg),
        axis=2,
    )
    out_data.append((gains, loss))

    return out_data


def boostrap_stylized_facts(
    data: npt.ArrayLike, B: int, S: int, L: int
) -> List[npt.NDArray]:
    data = np.asarray(data)
    assert data.ndim == 2

    out_data = []
    for fact, arg in zip(_stylized_fact_list, _stylized_fact_arg_list):
        t_data, _ = bootstrap.boostrap_distribution(data, fact, B, S, L, **arg)
        out_data.append(t_data)

    gains, loss = np.nanmean(
        gain_loss_asymetry.gain_loss_asymmetry(np.cumsum(data, axis=0), **_glarg),
        axis=2,
    )

    out_data.append((gains, loss))

    return out_data


def stylied_facts_from_model(sample_func, B: int, S: int) -> List[npt.NDArray]:
    out_data = []
    for fact, arg in zip(_stylized_fact_list, _stylized_fact_arg_list):
        b_strap = []
        for _ in range(B):
            data = sample_func(S)
            t_data = np.asarray(np.mean(fact(data, **arg), axis=1)).T
            b_strap.append(t_data)
        out_data.append(np.asarray(b_strap).T)

    data = sample_func(B)
    log_price = np.cumsum(data)
    gains, loss = np.nanmean(
        gain_loss_asymetry.gain_loss_asymmetry(log_price, max_lag=1000, theta=0.1),
        axis=2,
    )

    out_data.append((gains, loss))

    return out_data


def stylized_score(
    real_data_stf: List[npt.ArrayLike],
    syn_data_stf: List[npt.ArrayLike],
) -> List[Tuple[npt.ArrayLike, float]]:
    out_data = []

    # dont consider all data for the score as some data seems to make scores worse
    #                         lu       ht      vc       le      cf     gl
    score_lag_intervals = [(0, 1000), (0, 2), (0, 150), (0, 60), (1, 30), (0, 60)]

    # last fact is gain loss
    real_data_stf = real_data_stf.copy()
    real_gain, real_loss = real_data_stf.pop()
    syn_data_stf = syn_data_stf.copy()
    syn_gain, syn_loss = syn_data_stf.pop()
    gli = score_lag_intervals.pop()

    for real, syn, lagi in zip(real_data_stf, syn_data_stf, score_lag_intervals):
        n = real.shape[0]
        assert syn.shape[0] == n

        w_dist = []
        for i in range(lagi[0], lagi[1]):
            std = np.nanstd(real)
            w_dist.append(wasserstein_distance(real[i, :] / std, syn[i, :] / std))

        out_data.append((np.asarray(w_dist), np.mean(w_dist)))

    #######################
    # gain loss asymmetry
    #######################

    # comptue stds
    real_gain = real_gain[gli[0] : gli[1]]
    syn_gain = syn_gain[gli[0] : gli[1]]
    n_gains = np.sum(real_gain)
    mu_gains = np.dot(real_gain, np.arange(1, real_gain.size + 1)) / n_gains
    std_gains = np.sqrt(
        np.dot((np.arange(1, real_gain.size + 1) - mu_gains) ** 2, real_gain) / n_gains
    )

    real_loss = real_loss[gli[0] : gli[1]]
    syn_loss = syn_loss[gli[0] : gli[1]]
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


if __name__ == "__main__":
    import load_data

    sp500 = load_data.load_log_returns("sp500")
    smi = load_data.load_log_returns("smi")

    B = 50
    S = 24
    L = 8192

    fingan = train_fingan.load_fingan(
        "/home/nico/thesis/code/data/cache/results/epoch_43/model.pt"
    )

    def fingan_sampler(S):
        return train_fingan.sample_fingan(model=fingan, batch_size=S)

    garch_models = train_garch.load_garch(
        "/home/nico/thesis/code/data/cache/garch_experiments/GARCH_ged_2024_07_14-03_28_28/garch_models.pt"
    )

    def garch_sampler(S):
        with warnings.catch_warnings(action="ignore"):
            return train_garch.sample_garch(garch_models, S, L)

    real_stf = boostrap_stylized_facts(sp500, B, S, L)
    syn_stf = stylied_facts_from_model(fingan_sampler, B, S)
    mu_score, scores, _ = stylized_score(real_stf, syn_stf)

    print(scores, mu_score)

    syn_stf = stylied_facts_from_model(garch_sampler, B, S)
    mu_score, scores, _ = stylized_score(real_stf, syn_stf)
    print(scores, mu_score)

    # visualize_stylized_facts.visualize_stylized_facts(np.concatenate(syn_data,axis=1))
    # plt.show()

    # visualize_stylized_facts.visualize_stylized_facts(sp500)
    # plt.show()
