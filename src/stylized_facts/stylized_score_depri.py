from typing import Any, Dict, Literal

import coarse_fine_volatility
import gain_loss_asymetry
import heavy_tails
import leverage_effect
import linear_unpredictability
import load_data
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import volatility_clustering
from power_fit import fit_powerlaw


def stylized_score(log_returns: npt.ArrayLike, syn_log_returns: npt.ArrayLike):
    syn_stats = _compute_stats(syn_log_returns, "syn")
    real_stats = _compute_stats(log_returns, "real")
    return _stylized_score(**syn_stats, **real_stats)


def _compute_stats(log_returns: npt.ArrayLike, kind: Literal["real", "syn"]):
    log_prices = np.cumsum(log_returns, axis=0)
    stats = {}
    stats[f"lu_{kind}_stat"] = linear_unpredictability.linear_unpredictability_stats(
        log_returns, max_lag=1000
    )
    stats[f"ht_{kind}_stat"] = heavy_tails.heavy_tails_stats(
        log_returns, n_bins=1000, tail_quant=0.1
    )
    stats[f"vc_{kind}_stat"] = volatility_clustering.volatility_clustering_stats(
        log_returns, max_lag=1000
    )
    stats[f"le_{kind}_stat"] = leverage_effect.leverage_effect_stats(
        log_returns, max_lag=100
    )
    stats[f"cfv_{kind}_stat"] = coarse_fine_volatility.coarse_fine_volatility_stats(
        log_returns, tau=5, max_lag=100
    )
    stats[f"gl_{kind}_stat"] = gain_loss_asymetry.gain_loss_asymmetry_stat(
        log_prices, max_lag=1000, theta=0.1
    )

    return stats


def _stylized_score(
    lu_real_stat: Dict[str, Any],
    lu_syn_stat: Dict[str, Any],
    ht_real_stat: Dict[str, Any],
    ht_syn_stat: Dict[str, Any],
    vc_real_stat: Dict[str, Any],
    vc_syn_stat: Dict[str, Any],
    le_real_stat: Dict[str, Any],
    le_syn_stat: Dict[str, Any],
    cfv_real_stat: Dict[str, Any],
    cfv_syn_stat: Dict[str, Any],
    gl_real_stat: Dict[str, Any],
    gl_syn_stat: Dict[str, Any],
):
    EPS = 1e-9

    ###########################
    # LINEAR UNPREDICTABILITY #
    ###########################
    lurealmse, lu_real_mse_std = lu_real_stat["mse"], lu_real_stat["mse_std"]
    lu_syn_mse, lu_syn_mse_std = lu_syn_stat["mse"], lu_syn_stat["mse_std"]
    lu_score = [
        np.abs((lurealmse - lu_syn_mse) / (lu_real_mse_std + EPS)),
        np.abs(lu_syn_mse_std / (lu_real_mse_std + EPS) - 1),
    ]
    lu_score = np.nanmean(lu_score)

    ###############
    # HEAVY TAILS #
    ###############
    htreal = (
        ht_real_stat["pos_beta"],
        ht_real_stat["neg_beta"],
    )
    htrealstd = (
        ht_real_stat["pos_beta_std"],
        ht_real_stat["neg_beta_std"],
    )

    htsyn = (
        ht_syn_stat["pos_beta"],
        ht_syn_stat["neg_beta"],
    )

    ht_score = [
        np.abs((a - b) / (c + EPS)) for a, b, c in zip(htreal, htsyn, htrealstd)
    ]
    ht_score = np.nanmean(ht_score)

    #########################
    # VOLATILITY CLUSTERING #
    #########################
    vcreal = [vc_real_stat["beta"]]
    vcrealstd = [vc_real_stat["beta_std"]]
    vcsyn = [vc_syn_stat["beta"]]

    vc_score = [
        np.abs((a - b) / (c + EPS)) for a, b, c in zip(vcreal, vcsyn, vcrealstd)
    ]
    vc_score = np.nanmean(vc_score)

    ###################
    # LEVERAGE EFFECT #
    ###################
    lereal = [le_real_stat["beta"]]
    lerealstd = [le_real_stat["beta_std"]]

    lesyn = [le_syn_stat["beta"]]

    le_score = [np.abs((a - b) / c) for a, b, c in zip(lereal, lesyn, lerealstd)]
    le_score = np.nanmean(le_score)

    ##########################
    # COARSE FINE VOLATILITY #
    ##########################

    cfvreal = (cfv_real_stat["beta"], cfv_real_stat["argmin"])
    cfvrealstd = (
        cfv_real_stat["beta_std"],
        cfv_real_stat["argmin_std"],
    )

    cfvsyn = cfv_syn_stat["beta"], cfv_syn_stat["argmin"]

    cfv_score = [np.abs((a - b) / c) for a, b, c in zip(cfvreal, cfvsyn, cfvrealstd)]
    cfv_score = np.nanmean(cfv_score)

    #############################
    # GAIN LOSS ASYMMETRY SCORE #
    #############################
    gl_real = (gl_real_stat["arg_diff"],)
    gl_real_std = (gl_real_stat["arg_diff_std"],)

    gl_syn = [
        gl_syn_stat["arg_diff"],
    ]

    # asymmetric loss function, punish negative deviations stronger than postive ones
    # the factor is determined by the ratio of the extreme values
    asym = (gl_real_stat["arg_diff"] - gl_real_stat["arg_diff_min"]) / (
        gl_real_stat["arg_diff_max"] - gl_real_stat["arg_diff"]
    )
    gl_score = [
        (asym if (a - b) < 0 else 1) * np.abs((a - b) / c)
        for a, b, c in zip(gl_real, gl_syn, gl_real_std)
    ]
    gl_score = np.nanmean(gl_score)

    scores = {
        "lin unpred": lu_score,
        "heavy tails": ht_score,
        "vol cluster": vc_score,
        "lev effect": le_score,
        "cf vol": cfv_score,
        "gain loss": gl_score,
    }
    total_score = np.nanmean(list(scores.values()))

    return total_score, scores


if __name__ == "__main__":
    sp500 = load_data.load_log_returns("sp500")
    smi = load_data.load_log_returns("smi")

    sp500_stats = _compute_stats(sp500, "real")
    ht_pos_real, ht_neg_real = (
        sp500_stats["ht_real_stat"]["pos_beta"],
        sp500_stats["ht_real_stat"]["neg_beta"],
    )

    smi_lu_mse = np.mean(
        linear_unpredictability.linear_unpredictability(smi, max_lag=1000) ** 2
    )
    print(f"lin upred {smi_lu_mse}")

    # select the lower quantiles and compute fit
    tail_quant = 0.2
    pos_y, pos_x, neg_y, neg_x = heavy_tails.discrete_pdf(smi, n_bins=1000)
    pos_mask, neg_mask = np.cumsum(pos_y), np.cumsum(neg_y)
    pos_mask, neg_mask = (
        pos_mask > pos_mask[-1] * (1 - tail_quant) if pos_mask.size > 0 else [],
        neg_mask > neg_mask[-1] * (1 - tail_quant) if neg_mask.size > 0 else [],
    )
    _, _, _, pos_beta, _ = fit_powerlaw(
        pos_x[pos_mask], pos_y[pos_mask], optimize="none"
    )
    _, _, _, neg_beta, _ = fit_powerlaw(
        neg_x[neg_mask], neg_y[neg_mask], optimize="none"
    )
    tail_score = np.abs(pos_beta - ht_pos_real) + np.abs(neg_beta - ht_neg_real)
    print(f"tail score {tail_score}")

    smi_vol_clust = volatility_clustering.volatility_clustering(smi, max_lag=100)
    beta = sp500_stats["vc_real_stat"]["beta"]
    alpha = sp500_stats["vc_real_stat"]["alpha"]
    slope = np.power(np.arange(1, 101, 1), beta) * np.exp(alpha)

    vc_score = np.mean((np.mean(smi_vol_clust, axis=1) - slope) ** 2)

    vc_score_std_ = np.mean((smi_vol_clust - slope.reshape(-1, 1)) ** 2, axis=0)
    vc_score_std = np.std(vc_score_std_)
    vc_score_min = np.min(vc_score_std_)
    vc_score_max = np.max(vc_score_std_)

    print(f"vc score {vc_score}")
    print(f"vc std {vc_score_std}")
    print(f"vc min {vc_score_min}")
    print(f"vc max {vc_score_max}")

    smi_lev_eff = leverage_effect.leverage_effect(smi, max_lag=100)
    beta = sp500_stats["le_real_stat"]["beta"]
    alpha = sp500_stats["le_real_stat"]["alpha"]
    slope = np.power(np.arange(1, 101, 1), beta) * np.exp(alpha)

    plt.plot(-slope)
    plt.plot(np.mean(smi_lev_eff, axis=1))
    plt.show()

    vc_score = np.mean((np.mean(smi_lev_eff, axis=1) - slope) ** 2)
    vc_score_std_ = np.mean((smi_lev_eff - slope.reshape(-1, 1)) ** 2, axis=0)
    vc_score_std = np.std(vc_score_std_)
    vc_score_min = np.min(vc_score_std_)
    vc_score_max = np.max(vc_score_std_)

    print(f"le score {vc_score}")
    print(f"le std {vc_score_std}")
    print(f"le min {vc_score_min}")
    print(f"le max {vc_score_max}")
