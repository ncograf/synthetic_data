import time
from typing import Any, Dict, List, Tuple

import boosted_stats
import gen_inv_gamma
import load_data
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from scipy.stats import wasserstein_distance


def gain_loss_asymmetry_torch(log_price: torch.Tensor, max_lag: int, theta: float):
    """Gain Loss assymetry statisitcs EXTREMELY SLOW

    Compute the discrete pdf functions (histogramms for bins: 1, 2, 3, ..., max_lag)
    of the gains and losses statistics.

    Args:
        log_price (npt.ArrayLike): log prices
        max_lag (int): maximum lag to compute
        theta (float): threshold parameter

    Returns:
        Tuple[npt.NDArray, npt.NDArray]: gains_statistic, loss_statistic
    """

    numpy = False
    if not torch.is_tensor(log_price):
        numpy = True
        log_price = torch.tensor(log_price)

    t = time.time()
    log_price = log_price.nan_to_num(0, 0, 0)
    n = log_price.shape[0]
    cs = torch.cumsum(log_price, dim=0)
    cs = torch.nn.functional.pad(cs, (0, 0, 0, max_lag))
    cs_unfolded = cs.unfold(0, n, 1)  # max_lag x n_stocks x n
    cs = cs.T  # n_stocks x (n + max_lag)
    cs = cs.reshape(1, *cs.shape)[:, :, :-max_lag]  # 1 x n_stocks x n
    print(f"unfold and cumsum {time.time() - t}")

    log_p_incr = cs - cs_unfolded  # max_lag x n_stocks x n

    t = time.time()
    gain_values = torch.argmax((log_p_incr > theta).type(torch.int8), dim=0)
    print(f"argmax gains {time.time() - t}")
    t = time.time()
    loss_values = torch.argmax((-log_p_incr < -theta).type(torch.int8), dim=0)
    print(f"argmax losses {time.time() - t}")

    if numpy:
        gain_values = np.asarray(gain_values)
        loss_values = np.asarray(loss_values)

    return gain_values, loss_values


def gain_loss_asymmetry(
    log_price: npt.ArrayLike, max_lag: int, theta: float
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Gain Loss assymetry statisitcs

    Compute the discrete pdf functions (histogramms for bins: 1, 2, 3, ..., max_lag)
    of the gains and losses statistics.

    Args:
        log_price (npt.ArrayLike): log prices
        max_lag (int): maximum lag to compute
        theta (float): threshold parameter

    Returns:
        Tuple[npt.NDArray, npt.NDArray]: gains_statistic, loss_statistic
    """

    log_price = np.asarray(log_price)
    if log_price.ndim == 1:
        log_price = log_price.reshape((-1, 1))
    elif log_price.ndim > 2:
        raise RuntimeError(
            f"Log Price has {log_price.ndim} dimensions should have 1 or 2."
        )

    if log_price.dtype.name == "float32":
        boosted = boosted_stats.gain_loss_asym_float(log_price, max_lag, theta, False)
    elif log_price.dtype.name == "float64":
        boosted = boosted_stats.gain_loss_asym_double(log_price, max_lag, theta, False)
    else:
        raise ValueError(f"Unsupported data type: {log_price.dtype.name}")

    boosted_gain = boosted[0] / (np.nansum(boosted[0], axis=0) + 1e-10)
    boosted_loss = boosted[1] / (np.nansum(boosted[1], axis=0) + 1e-10)
    return boosted_gain, boosted_loss


def gain_loss_asymmetry_stat(
    log_price: npt.ArrayLike, max_lag: int, theta: float
) -> Dict[str, Any]:
    """Gain Loss asymmetry statistics

    Args:
        log_price (npt.ArrayLike): log prices
        max_lag (int): maximum lag to compute
        theta (float): threshold parameter

    Returns:
        Dict[str, Any]: result dictonary with keys:
            vol_clust: volatility clustering data (max_lag x stocks)
            power_fit_x: powerlaw x values used for fit
            corr: pearson correlation coefficient of powerlaw fit
            rate: exponent fitted in powerlaw
            const : constant fitted in powerlaw
            corr_std: standard deviation for fits
            rate_std: standard deviation for fits
            const_std: standard deviation for fits
    """

    gain, loss = gain_loss_asymmetry(log_price=log_price, max_lag=max_lag, theta=theta)

    gain_avg = np.nanmean(gain, axis=1)
    loss_avg = np.nanmean(loss, axis=1)
    gain_order = np.flip(np.argsort(gain_avg))
    loss_order = np.flip(np.argsort(loss_avg))
    arg_max_gain = np.nanmedian(gain_order[:10])
    arg_max_loss = np.nanmedian(loss_order[:10])
    max_gain = np.nanmean(gain_avg[gain_order[:10]])
    max_loss = np.nanmean(loss_avg[loss_order[:10]])
    diff = max_loss - max_gain
    arg_diff = arg_max_gain - arg_max_loss

    # variace estimation
    max_gain_arr, max_loss_arr, arg_max_gain_arr, arg_max_loss_arr = [], [], [], []
    for idx in range(gain.shape[1]):
        go = np.flip(np.argsort(gain[:, idx]))
        lo = np.flip(np.argsort(loss[:, idx]))
        amg = np.nanmedian(go[:10])
        aml = np.nanmedian(lo[:10])
        mg = np.nanmean(gain[go[:10]])
        ml = np.nanmean(loss[lo[:10]])
        max_gain_arr.append(mg)
        max_loss_arr.append(ml)
        arg_max_gain_arr.append(amg)
        arg_max_loss_arr.append(aml)

    std_max_gain = np.nanstd(max_gain_arr)
    std_max_loss = np.nanstd(max_loss_arr)
    arg_std_max_gain = np.nanstd(arg_max_gain_arr)
    arg_std_max_loss = np.nanstd(arg_max_loss_arr)
    std_arg_diff = np.nanstd(np.array(arg_max_gain_arr) - np.array(arg_max_loss_arr))
    arg_diff_min = np.nanmin(np.array(arg_max_gain_arr) - np.array(arg_max_loss_arr))
    arg_diff_max = np.nanmax(np.array(arg_max_gain_arr) - np.array(arg_max_loss_arr))
    arg_diff_mean = np.nanmean(np.array(arg_max_gain_arr) - np.array(arg_max_loss_arr))
    arg_diff_median = np.nanmedian(
        np.array(arg_max_gain_arr) - np.array(arg_max_loss_arr)
    )
    std_diff = np.nanstd(np.array(max_loss_arr) - np.array(max_gain_arr))

    stats = {
        "gain": gain,
        "loss": loss,
        "arg_max_gain": arg_max_gain,
        "arg_max_loss": arg_max_loss,
        "max_gain": max_gain,
        "max_loss": max_loss,
        "diff": diff,
        "arg_diff": arg_diff,
        "arg_max_gain_std": arg_std_max_gain,
        "arg_max_loss_std": arg_std_max_loss,
        "max_gain_std": std_max_gain,
        "max_loss_std": std_max_loss,
        "diff_std": std_diff,
        "arg_diff_std": std_arg_diff,
        "arg_diff_min": arg_diff_min,
        "arg_diff_max": arg_diff_max,
        "arg_diff_mean": arg_diff_mean,
        "arg_diff_median": arg_diff_median,
    }

    return stats


def visualize_stat(
    plot: plt.Axes,
    log_price: npt.NDArray,
    name: str,
    print_stats: List[str],
    fit: bool = True,
):
    stat = gain_loss_asymmetry_stat(log_price=log_price, max_lag=1000, theta=0.1)
    gain, loss = stat["gain"], stat["loss"]
    gain_data = np.mean(gain, axis=1)
    loss_data = np.mean(loss, axis=1)

    if fit:
        theta_gain, log_like_gain = gen_inv_gamma.fit_gen_inv_gamma(
            gain_data, [11.2, 2.4, 0.5, 4.5], method="Newton-CG"
        )
        theta_loss, log_like_loss = gen_inv_gamma.fit_gen_inv_gamma(
            loss_data, [11.2, 2.4, 0.5, 4.5], method="Newton-CG"
        )
        x_lin = np.linspace(1, 1000, num=3000)

    gain_loss_axis_setting = {
        "title": f"{name} gain loss asymetry",
        "ylabel": r"return time probability",
        "xlabel": "lag k in days",
        "xscale": "log",
        "yscale": "linear",
    }

    gain_plot_setting = {
        "alpha": 0.8,
        "marker": "o",
        # "color": "violet",
        "markersize": 3,
        "linestyle": "None",
        "label": r"gain $\theta > 0$",
    }

    loss_plot_setting = {
        "alpha": 0.8,
        "marker": "o",
        # "color": "cornflowerblue",
        "markersize": 3,
        "linestyle": "None",
        "label": r"loss $\theta < 0$",
    }

    plot.plot(gain_data, **gain_plot_setting)
    plot.plot(loss_data, **loss_plot_setting)

    max_gain, max_loss, arg_max_gain, arg_max_loss = (
        stat["max_gain"],
        stat["max_loss"],
        stat["arg_max_gain"],
        stat["arg_max_loss"],
    )

    if fit:
        y_lin = gen_inv_gamma.gen_inf_gamma_pdf(x_lin, theta_gain)
        plot.plot(
            x_lin, y_lin, linestyle="--", alpha=0.8, color="red", label="Fit gains"
        )
    plot.plot(
        [arg_max_gain, arg_max_gain],
        [0, max_gain],
        color="red",
        linestyle=":",
        linewidth=2,
        label="gain peak",
    )
    if fit:
        y_lin = gen_inv_gamma.gen_inf_gamma_pdf(x_lin, theta_loss)
        plot.plot(
            x_lin, y_lin, linestyle="--", alpha=1, color="blue", label="Fit losses"
        )
    plot.plot(
        [arg_max_loss, arg_max_loss],
        [0, max_loss],
        color="navy",
        linestyle=":",
        linewidth=2,
        label="loss peak",
    )

    if fit:
        stat["gain_likelyhood"] = np.exp(-log_like_gain)
        stat["loss_likelyhood"] = np.exp(-log_like_loss)
    for key in print_stats:
        print(f"{name} gain-loss-asym {key} {stat[key]}")

    plot.set(**gain_loss_axis_setting)
    plot.legend(loc="upper right")


if __name__ == "__main__":
    sp500 = np.cumsum(load_data.load_log_returns("sp500"), axis=0)
    smi = np.cumsum(load_data.load_log_returns("smi"), axis=0)
    dax = np.cumsum(load_data.load_log_returns("dax"), axis=0)
    # visualize_stat(plt.gca(), data, 'test', 0.1, interval=True)

    lag = 60
    sp500 = np.mean(gain_loss_asymmetry(sp500, max_lag=lag, theta=0.1), axis=-1)
    smi = np.mean(gain_loss_asymmetry(smi, max_lag=lag, theta=0.1), axis=-1)
    dax = np.mean(gain_loss_asymmetry(dax, max_lag=lag, theta=0.1), axis=-1)

    values = np.arange(1, lag + 2, 1) / lag
    dax_smi = [wasserstein_distance(values, dax[i, :], smi[i, :]) for i in [0, 1]]
    sp_dax = [
        wasserstein_distance(values, values, sp500[i, :], dax[i, :]) for i in [0, 1]
    ]
    sp_smi = [
        wasserstein_distance(values, values, sp500[i, :], smi[i, :]) for i in [0, 1]
    ]

    print(wasserstein_distance(values, values, sp500[0, :], sp500[1, :]))

    plt.plot(dax_smi, label=f"dax smi {dax_smi}", linestyle="none", marker="o")
    plt.plot(sp_dax, label=f"sp500 dax {sp_dax}", linestyle="none", marker="o")
    plt.plot(sp_smi, label=f"sp500 smi {sp_smi}", linestyle="none", marker="o")
    plt.legend()
    plt.show()

    sp500 = np.cumsum(load_data.load_log_returns("sp500"), axis=0)
    smi = np.cumsum(load_data.load_log_returns("smi"), axis=0)
    dax = np.cumsum(load_data.load_log_returns("dax"), axis=0)

    visualize_stat(plt.gca(), sp500, "test", [], False)
    visualize_stat(plt.gca(), dax, "test", [], False)
    visualize_stat(plt.gca(), smi, "test", [], False)
    plt.show()

    # t = time.time()
    # test = gain_loss_asymmetry_torch(data, max_lag=1000, theta=0.1)
    # print("torch ", time.time() - t)

    # t = time.time()
    # test = gain_loss_asymmetry(data, max_lag=1000, theta=0.1)
    # print("normal ", time.time() - t)

    # print(test[0].shape)
