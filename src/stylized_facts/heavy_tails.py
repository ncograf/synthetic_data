from typing import Any, Dict

import bootstrap
import load_data
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from power_fit import fit_powerlaw


def discrete_pdf(
    log_returns: torch.Tensor, n_bins: int | None = None
) -> torch.Tensor | npt.NDArray:
    if n_bins is None:
        n_bins = np.maximum(np.minimum(2000, np.asarray(log_returns).size // 400), 10)

    numpy = False
    if not torch.is_tensor(log_returns):
        numpy = True
        log_returns = torch.tensor(log_returns)

    # distribution flattens log returns and remves nans
    log_returns = log_returns.nan_to_num(0, 0, 0)
    log_returns = log_returns.flatten()
    log_returns = log_returns / torch.std(log_returns)

    min, mu, max = (
        torch.min(log_returns),
        torch.mean(log_returns),
        torch.max(log_returns),
    )
    exp = 4
    pos_bin_end = torch.pow(
        torch.linspace(
            1e-3 ** (1 / exp), torch.pow(max - mu, 1 / exp), n_bins // 2 + 1
        ),
        exp,
    )[int(n_bins * exp / 100) :]
    pos_diffs = torch.diff(pos_bin_end)
    neg_bin_end = torch.pow(
        torch.linspace(
            1e-3 ** (1 / exp), torch.pow(mu - min, 1 / exp), n_bins // 2 + 1
        ),
        exp,
    )[int(n_bins * exp / 100) :]
    neg_diffs = torch.diff(neg_bin_end)

    pos_log_ret = log_returns[log_returns > mu] - mu
    neg_log_ret = -(log_returns[log_returns <= mu] - mu)

    pos_log_ret = torch.sort(pos_log_ret)[0]
    neg_log_ret = torch.sort(neg_log_ret)[0]

    pos_cum_cnt = torch.searchsorted(pos_log_ret, pos_bin_end)
    neg_cum_cnt = torch.searchsorted(neg_log_ret, neg_bin_end)

    pos_cum_cnt = pos_cum_cnt[1:] - pos_cum_cnt[:-1]
    neg_cum_cnt = neg_cum_cnt[1:] - neg_cum_cnt[:-1]

    pos_dens = pos_cum_cnt / (log_returns.numel() * pos_diffs)
    neg_dens = neg_cum_cnt / (log_returns.numel() * neg_diffs)

    if numpy:
        out = (
            np.asarray(pos_dens),
            np.asarray(pos_bin_end[1:]),
            np.asarray(neg_dens),
            np.asarray(neg_bin_end[1:]),
        )
    else:
        out = pos_dens, pos_bin_end, neg_dens, neg_bin_end

    return out


def heavy_tails(dens: npt.ArrayLike, bins: npt.ArrayLike, tq: float):
    """Compute tail slope of the empirical distribution

    Args:
        dens (npt.ArrayLike): densities
        bins (npt.ArrayLike): bin ends
        tq (float): tail quantile to be considered

    Returns:
        ndarray: slope, intercept
    """

    dens, bins = np.asarray(dens), np.asarray(bins)

    p_mask = np.cumsum(dens)
    p_mask = p_mask > p_mask[-1] * (1 - tq) if p_mask.size > 0 else []
    p_mask = p_mask & (dens > 0)

    pX = np.log(bins[p_mask])
    pX = np.stack([pX, np.ones_like(pX)], axis=1)
    pY = np.log(dens[p_mask])
    p = np.linalg.lstsq(pX, pY, rcond=None)[0]

    return p, bins[p_mask]


def heavy_tails_stats(
    log_returns: npt.ArrayLike, n_bins: int, tail_quant: float
) -> Dict[str, Any]:
    """Heavy tails statistics

    Args:
        log_returns (npt.ArrayLike): log returns
        n_bins (int): number of bins for histogram
        tail_quant (float): quantile of the tails to be considered in the fit

    Returns:
        Dict[str, Any]: result dictonary with keys:
            pos_dens: heavy_tails data positive probabilities (n_bins x stocks)
            pos_bins: heavy_tails data positive bins (n_bins x stocks)
            pos_powerlaw_x: positive powerlaw x values used for fit
            pos_corr: pearson correlation coefficient of powerlaw fit
            pos_beta: exponent fitted in powerlaw
            pos_alpha : constant fitted in powerlaw
            pos_beta_std: standard deviation for positive fits
            pos_alpha_std: standard deviation for positive fits
            pos_corr_std: standard deviation for positive fits
            neg_dens: heavy_tails data negative probabilities (n_bins x stocks)
            neg_bins: heavy_tails data negative bins (n_bins x stocks)
            neg_powerlaw_x: negative powerlaw x values used for fit
            neg_corr: pearson correlation coefficient of powerlaw fit
            neg_beta: exponent fitted in powerlaw
            neg_alpha : constant fitted in powerlaw
            neg_std: mean squared error standard deviation over stocks
            neg_beta_std: standard deviation for negative fits
            neg_alpha_std: standard deviation for negative fits
            neg_corr_std: standard deviation for negative fits
    """

    pos_y, pos_x, neg_y, neg_x = discrete_pdf(log_returns=log_returns, n_bins=n_bins)
    pos_mask, neg_mask = np.cumsum(pos_y), np.cumsum(neg_y)
    pos_mask, neg_mask = (
        pos_mask > pos_mask[-1] * (1 - tail_quant) if pos_mask.size > 0 else [],
        neg_mask > neg_mask[-1] * (1 - tail_quant) if neg_mask.size > 0 else [],
    )

    pos_fit_x, _, pos_alpha, pos_beta, pos_r = fit_powerlaw(
        pos_x[pos_mask], pos_y[pos_mask], optimize="none"
    )
    neg_fit_x, _, neg_alpha, neg_beta, neg_r = fit_powerlaw(
        neg_x[neg_mask], neg_y[neg_mask], optimize="none"
    )

    # variace estimation
    pos_y_arr, pos_x_arr, neg_y_arr, neg_x_arr = [], [], [], []
    for i in range(log_returns.shape[1]):
        py, px, ny, nx = discrete_pdf(log_returns[:, i], n_bins // 40)
        pos_mask, neg_mask = np.cumsum(py), np.cumsum(ny)
        pos_mask, neg_mask = (
            pos_mask > pos_mask[-1] * (1 - tail_quant) if pos_mask.size > 0 else [],
            neg_mask > neg_mask[-1] * (1 - tail_quant) if neg_mask.size > 0 else [],
        )
        pos_y_arr.append(py[pos_mask])
        pos_x_arr.append(px[pos_mask])
        neg_y_arr.append(ny[neg_mask])
        neg_x_arr.append(nx[neg_mask])

    pos_alpha_arr, pos_beta_arr, pos_r_arr = [], [], []
    for px, py in zip(pos_x_arr, pos_y_arr):
        _, _, a, b, r = fit_powerlaw(px, py, optimize="none")
        pos_alpha_arr.append(a)
        pos_beta_arr.append(b)
        pos_r_arr.append(r)

    neg_alpha_arr, neg_beta_arr, neg_r_arr = [], [], []
    for px, py in zip(neg_x_arr, neg_y_arr):
        _, _, a, b, r = fit_powerlaw(px, py, optimize="none")
        neg_alpha_arr.append(a)
        neg_beta_arr.append(b)
        neg_r_arr.append(r)

    stats = {
        "pos_dens": pos_y,
        "pos_bins": pos_x,
        "pos_powerlaw_x": pos_fit_x,
        "pos_corr": pos_r,
        "pos_beta": pos_beta,
        "pos_alpha": pos_alpha,
        "pos_alpha_std": np.nanstd(pos_alpha_arr),
        "pos_corr_std": np.nanstd(pos_r_arr),
        "pos_beta_std": np.nanstd(pos_beta_arr),
        "pos_beta_max": np.nanmax(pos_beta_arr),
        "pos_beta_min": np.nanmin(pos_beta_arr),
        "pos_beta_mean": np.nanmean(pos_beta_arr),
        "pos_beta_median": np.nanmedian(pos_beta_arr),
        "neg_dens": neg_y,
        "neg_bins": neg_x,
        "neg_powerlaw_x": neg_fit_x,
        "neg_corr": neg_r,
        "neg_beta": neg_beta,
        "neg_alpha": neg_alpha,
        "neg_alpha_std": np.nanstd(neg_alpha_arr),
        "neg_corr_std": np.nanstd(neg_r_arr),
        "neg_beta_std": np.nanstd(neg_beta_arr),
        "neg_beta_max": np.nanmax(neg_beta_arr),
        "neg_beta_min": np.nanmin(neg_beta_arr),
        "neg_beta_mean": np.nanmean(neg_beta_arr),
        "neg_beta_median": np.nanmedian(neg_beta_arr),
    }

    return stats


def visualize_boostrap(plot: plt.Axes, b_dist: npt.NDArray, x_lim: npt.NDArray):
    B = b_dist.shape[1]
    ninty_five = np.array([B * 0.025, B * 0.975]).astype(int)
    ninty_five = b_dist[:, ninty_five]

    lin_x = np.linspace(x_lim[0], x_lim[1], num=10)

    # compute positive fits
    pos_y_high = np.exp(ninty_five[1, 0]) * np.power(lin_x, ninty_five[0, 0])
    pos_y_low = np.exp(ninty_five[1, 1]) * np.power(lin_x, ninty_five[0, 1])
    plot.fill_between(lin_x, pos_y_high, pos_y_low, fc="navy", alpha=0.1)

    # compute negative fits
    neg_y_high = np.exp(ninty_five[3, 0]) * np.power(lin_x, ninty_five[2, 0])
    neg_y_low = np.exp(ninty_five[3, 1]) * np.power(lin_x, ninty_five[2, 1])
    plot.fill_between(lin_x, neg_y_high, neg_y_low, fc="red", alpha=0.1)


def visualize_stat(
    plot: plt.Axes,
    log_returns: npt.NDArray,
    name: str,
    tq: float = 0.1,
    interval: bool = False,
):
    # compute distribution
    pos_y, pos_x, neg_y, neg_x = discrete_pdf(log_returns)

    # plot data
    plot.set(
        title=f"{name} heavy tails",
        ylabel=r"density $P\left(\tilde{r_t}\right)$",
        xlabel=r"normalized returns $\tilde{r_t} := r_t\, /\, \sigma$",
        xscale="log",
        yscale="log",
    )
    plot.plot(
        pos_x,
        pos_y,
        alpha=0.8,
        marker="o",
        color="violet",
        markersize=2,
        linestyle="None",
        label=r"pos. $\tilde{r}_t > 0$",
    )
    plot.plot(
        neg_x,
        neg_y,
        alpha=0.8,
        marker="o",
        color="cornflowerblue",
        markersize=2,
        linestyle="None",
        label=r"neg. $\tilde{r}_t < 0$",
    )
    xlim = plot.get_xlim()
    ylim = plot.get_ylim()

    # compute positive fits
    (pos_beta, pos_alpha), pos_x_fit = heavy_tails(pos_y, pos_x, tq=tq)
    pos_x_lin = np.linspace(np.min(pos_x_fit), np.max(pos_x_fit), num=100)
    pos_y_lin = np.exp(pos_alpha) * np.power(pos_x_lin, pos_beta)

    # adjust the lines to fit the plot
    pos_filter = (pos_y_lin > ylim[0]) & (pos_y_lin < ylim[1])
    pos_x_lin = pos_x_lin[pos_filter][:-15]
    pos_y_lin = pos_y_lin[pos_filter][:-15]

    # compute negative fit
    (neg_beta, neg_alpha), neg_x_fit = heavy_tails(neg_y, neg_x, tq=tq)
    neg_x_lin = np.linspace(np.min(neg_x_fit), np.max(neg_x_fit), num=100)
    neg_y_lin = np.exp(neg_alpha) * np.power(neg_x_lin, neg_beta)

    # adjust the lines to fit the plot
    neg_filter = (neg_y_lin > ylim[0]) & (neg_y_lin < ylim[1])
    neg_x_lin = neg_x_lin[neg_filter][:-15]
    neg_y_lin = neg_y_lin[neg_filter][:-15]

    # plot the fitted lines
    plot.plot(
        pos_x_lin,
        pos_y_lin,
        label=f"pos. $p(\\tilde{{r}}_t) \\propto \\tilde{{r}}_t^{{{pos_beta:.2f}}}$",
        linewidth=2,
        linestyle="--",
        alpha=1,
        color="red",
    )
    plot.plot(
        neg_x_lin,
        neg_y_lin,
        label=f"neg. $p(\\tilde{{r}}_t) \\propto \\tilde{{r}}_t^{{{neg_beta:.2f}}}$",
        linewidth=2,
        linestyle="--",
        alpha=1,
        color="navy",
    )
    plot.set_xlim(xlim)
    plot.set_ylim(ylim)
    plot.legend(loc="lower left")

    if interval:

        def stf(data):
            pd, pb, nd, nb = discrete_pdf(data)
            (pb, pa), _ = heavy_tails(pd, pb, tq)
            (nb, na), _ = heavy_tails(nd, nb, tq)

            return np.array([pb, pa, nb, na]).reshape(-1, 1)

        b_dist, _ = bootstrap.boostrap_distribution(
            log_returns, stf, B=500, S=24, L=4096
        )
        x_lim = (np.min(pos_x_lin), np.max(pos_x_lin))

        visualize_boostrap(plot, b_dist, x_lim)


if __name__ == "__main__":
    sp500 = load_data.load_log_returns("sp500")
    smi = load_data.load_log_returns("smi")
    dax = load_data.load_log_returns("dax")
    # visualize_stat(plt.gca(), data, 'test', 0.1, interval=True)

    def stf(data):
        pd, pb, nd, nb = discrete_pdf(data)
        (pb, pa), _ = heavy_tails(pd, pb, 0.1)
        (nb, na), _ = heavy_tails(nd, nb, 0.1)

        return np.array([pb, pa, nb, na]).reshape(-1, 1)

    from scipy.stats import wasserstein_distance

    B = 500
    S = 24
    L = 4096
    stf_b_sp500, stf_sp500 = bootstrap.boostrap_distribution(sp500, stf, B, S, L)
    stf_b_smi, stf_smi = bootstrap.boostrap_distribution(smi, stf, B, S, L)
    stf_b_dax, stf_dax = bootstrap.boostrap_distribution(dax, stf, B, S, L)

    sp_smi = []
    sp_dax = []
    dax_smi = []

    fact = 0.1
    for i in range(stf_b_dax.shape[0]):
        dax_smi.append(
            wasserstein_distance(stf_b_dax[i, :] * fact, stf_b_smi[i, :] * fact)
        )
        sp_dax.append(
            wasserstein_distance(stf_b_dax[i, :] * fact, stf_b_sp500[i, :] * fact)
        )
        sp_smi.append(
            wasserstein_distance(stf_b_smi[i, :] * fact, stf_b_sp500[i, :] * fact)
        )

    plt.plot(dax_smi, label=f"dax smi {np.mean(dax_smi)}", linestyle="none", marker="o")
    plt.plot(sp_dax, label=f"sp500 dax {np.mean(sp_dax)}", linestyle="none", marker="o")
    plt.plot(sp_smi, label=f"sp500 smi {np.mean(sp_smi)}", linestyle="none", marker="o")
    plt.legend()
    plt.show()
