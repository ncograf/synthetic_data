from typing import Any, Dict, Literal

import coarse_fine_volatility
import gain_loss_asymetry
import heavy_tails
import leverage_effect
import linear_unpredictability
import numpy as np
import numpy.typing as npt
import volatility_clustering


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
    stats[f"ht_{kind}_stat"] = heavy_tails.heavy_tails_stats(log_returns, n_bins=1000)
    stats[f"vc_{kind}_stat"] = volatility_clustering.volatility_clustering_stats(
        log_returns, max_lag=1000
    )
    stats[f"le_{kind}_stat"] = leverage_effect.leverage_effect_stats(
        log_returns, max_lag=100
    )
    stats[f"cfv_{kind}_stat"] = coarse_fine_volatility.coarse_fine_volatility_stats(
        log_returns, tau=5, max_lag=30
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
        ((lurealmse - lu_syn_mse) / (lu_real_mse_std + EPS)) ** 2,
        (lu_syn_mse_std / (lu_real_mse_std + EPS) - 1) ** 2,
    ]
    lu_score = np.mean(lu_score)

    ###############
    # HEAVY TAILS #
    ###############
    htreal = (
        ht_real_stat["pos_rate"],
        ht_real_stat["neg_rate"],
        ht_real_stat["pos_corr"],
        ht_real_stat["neg_corr"],
    )
    htrealstd = (
        ht_real_stat["pos_rate_std"],
        ht_real_stat["neg_rate_std"],
        ht_real_stat["pos_corr_std"],
        ht_real_stat["neg_corr_std"],
    )

    htsyn = (
        ht_syn_stat["pos_rate"],
        ht_syn_stat["neg_rate"],
        ht_syn_stat["pos_corr"],
        ht_syn_stat["neg_corr"],
    )
    htsynstd = (
        ht_syn_stat["pos_rate_std"],
        ht_syn_stat["neg_rate_std"],
        ht_syn_stat["pos_corr_std"],
        ht_syn_stat["neg_corr_std"],
    )

    ht_score = [((a - b) / (c + EPS)) ** 2 for a, b, c in zip(htreal, htsyn, htrealstd)]
    ht_score += [(b / (a + EPS) - 1) ** 2 for a, b in zip(htrealstd, htsynstd)]
    ht_score = np.mean(ht_score)

    #########################
    # VOLATILITY CLUSTERING #
    #########################
    vcreal = vc_real_stat["rate"], vc_real_stat["corr"]
    vcrealstd = vc_real_stat["rate_std"], vc_real_stat["corr_std"]

    vcsyn = vc_syn_stat["rate"], vc_syn_stat["corr"]
    vcsynstd = vc_syn_stat["rate_std"], vc_syn_stat["corr_std"]

    vc_score = [((a - b) / (c + EPS)) ** 2 for a, b, c in zip(vcreal, vcsyn, vcrealstd)]
    vc_score += [(b / (a + EPS) - 1) ** 2 for a, b in zip(vcrealstd, vcsynstd)]
    vc_score = np.mean(vc_score)

    ###################
    # LEVERAGE EFFECT #
    ###################
    lereal = le_real_stat["pow_rate"], le_real_stat["pow_r"]
    lerealstd = le_real_stat["pow_rate_std"], le_real_stat["pow_r_std"]

    lesyn = le_syn_stat["pow_rate"], le_syn_stat["pow_r"]
    lesynstd = le_syn_stat["pow_rate_std"], le_syn_stat["pow_r_std"]

    le_score = [((a - b) / c) ** 2 for a, b, c in zip(lereal, lesyn, lerealstd)]
    le_score += [(b / (a + EPS) - 1) ** 2 for a, b in zip(lerealstd, lesynstd)]
    le_score = np.mean(le_score)

    ##########################
    # COARSE FINE VOLATILITY #
    ##########################

    cfvreal = cfv_real_stat["beta"], cfv_real_stat["alpha"], cfv_real_stat["argmin"]
    cfvrealstd = (
        cfv_real_stat["beta_std"],
        cfv_real_stat["alpha_std"],
        cfv_real_stat["argmin_std"],
    )

    cfvsyn = cfv_syn_stat["beta"], cfv_syn_stat["alpha"], cfv_syn_stat["argmin"]
    cfvsynstd = (
        cfv_syn_stat["beta_std"],
        cfv_syn_stat["alpha_std"],
        cfv_syn_stat["argmin_std"],
    )

    cfv_score = [((a - b) / c) ** 2 for a, b, c in zip(cfvreal, cfvsyn, cfvrealstd)]
    cfv_score += [(b / (a + EPS) - 1) ** 2 for a, b in zip(vcrealstd, cfvsynstd)]
    cfv_score = np.mean(cfv_score)

    #############################
    # GAIN LOSS ASYMMETRY SCORE #
    #############################
    gl_real = (
        gl_real_stat["arg_max_gain"],
        gl_real_stat["arg_max_loss"],
        gl_real_stat["max_gain"],
        gl_real_stat["max_loss"],
    )
    gl_real_std = (
        gl_real_stat["std_arg_max_gain"],
        gl_real_stat["std_arg_max_loss"],
        gl_real_stat["std_max_gain"],
        gl_real_stat["std_max_loss"],
    )

    gl_syn = [
        gl_syn_stat["arg_max_gain"],
        gl_syn_stat["arg_max_loss"],
        gl_syn_stat["max_gain"],
        gl_syn_stat["max_loss"],
    ]
    gl_syn_std = (
        gl_syn_stat["std_arg_max_gain"],
        gl_syn_stat["std_arg_max_loss"],
        gl_syn_stat["std_max_gain"],
        gl_syn_stat["std_max_loss"],
    )

    gl_score = [((a - b) / c) ** 2 for a, b, c in zip(gl_real, gl_syn, gl_real_std)]
    gl_score += [(b / (a + EPS) - 1) ** 2 for a, b in zip(gl_real_std, gl_syn_std)]
    gl_score = np.mean(gl_score)

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
