import coarse_fine_volatility
import gain_loss_asymetry
import heavy_tails
import leverage_effect
import linear_unpredictability
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import volatility_clustering


def visualize_stylized_facts_paper(
    log_returns: npt.ArrayLike, textwidth: float
) -> plt.Figure:
    """Visualizes all stilized facts and returns the plt figure

    Args:
        log_returns (array_like):  (n_timesteps x m_stocks) or (n_timesteps) return data.
        textwidth (float): Textwith to create figure for

    Returns:
        plt.Figure: matplotlib figure ready to plot / save
    """

    # configure plt plots
    figure_style = {
        "text.usetex": True,
        "pgf.preamble": r"\usepackage{amsmath}",
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
    }
    subplot_layout = {
        "ncols": 3,
        "nrows": 2,
        "sharex": "none",
        "sharey": "none",
    }

    matplotlib.use("pgf")
    plt.rcParams.update(figure_style)
    plt.style.use(["science", "ieee"])

    aspect_ratio = 10 / 16
    scale = 1.0
    width = textwidth * scale
    height = width * aspect_ratio
    fig, axes = plt.subplots(
        **subplot_layout, constrained_layout=True, figsize=(width, height), dpi=600
    )

    # prepare data
    log_returns = np.asarray(log_returns)
    if log_returns.ndim == 1:
        log_returns = log_returns.reshape((-1, 1))
    elif log_returns.ndim > 2:
        raise RuntimeError(f"Log Returns has {log_returns.ndim} dimensions.")

    log_price = np.cumsum(log_returns, axis=0)

    line_size = 1
    emph_size = 1.2

    lu_data = linear_unpredictability.linear_unpredictability_stats(
        log_returns=log_returns, max_lag=1000
    )
    lu_data = np.mean(lu_data["data"], axis=1)
    axes[0, 0].set(
        title="linear unpredictability",
        ylabel=r"$Corr(r_t, r_{t+k})$",
        xlabel=r"lag $k$",
        xscale="log",
        yscale="linear",
        ylim=(-1, 1),
    )
    axes[0, 0].plot(
        lu_data,
        marker=".",
        color="cornflowerblue",
        markersize=line_size,
        linestyle="None",
    )

    # compute statistics
    stat = heavy_tails.heavy_tails_stats(
        log_returns=log_returns, n_bins=1000, tail_quant=0.1
    )
    pos_x, pos_y, pos_fit_x = stat["pos_bins"], stat["pos_dens"], stat["pos_powerlaw_x"]
    neg_x, neg_y, neg_fit_x = stat["neg_bins"], stat["neg_dens"], stat["neg_powerlaw_x"]

    # plot data
    plot = axes[0, 1]
    plot.set(
        title="heavy tails",
        ylabel=r"pdf $P\left(\tilde{r_t}\right)$",
        xlabel=r"$\tilde{r_t} := r_t\, /\, \sigma$",
        xscale="log",
        yscale="log",
    )
    plot.plot(
        pos_x,
        pos_y,
        alpha=0.8,
        marker=".",
        color="violet",
        markersize=line_size,
        linestyle="None",
    )
    plot.plot(
        neg_x,
        neg_y,
        alpha=0.8,
        marker=".",
        color="cornflowerblue",
        markersize=line_size,
        linestyle="None",
    )
    xlim = plot.get_xlim()
    ylim = plot.get_ylim()

    # compute positive fits
    pos_alpha, pos_beta = stat["pos_alpha"], stat["pos_beta"]
    pos_x_lin = np.linspace(np.min(pos_fit_x), np.max(pos_fit_x), num=1000)
    pos_y_lin = np.exp(pos_alpha) * np.power(pos_x_lin, pos_beta)

    # adjust the lines to fit the plot
    pos_filter = (pos_y_lin > ylim[0]) & (pos_y_lin < ylim[1])
    pos_x_lin = pos_x_lin[pos_filter][:-100]
    pos_y_lin = pos_y_lin[pos_filter][:-100]

    # compute negative fit
    neg_alpha, neg_beta = stat["neg_alpha"], stat["neg_beta"]
    neg_x_lin = np.linspace(np.min(neg_fit_x), np.max(neg_fit_x), num=1000)
    neg_y_lin = np.exp(neg_alpha) * np.power(neg_x_lin, neg_beta)

    # adjust the lines to fit the plot
    neg_filter = (neg_y_lin > ylim[0]) & (neg_y_lin < ylim[1])
    neg_x_lin = neg_x_lin[neg_filter][:-100]
    neg_y_lin = neg_y_lin[neg_filter][:-100]

    # plot the fitted lines
    plot.plot(
        pos_x_lin,
        pos_y_lin,
        linewidth=emph_size,
        linestyle="--",
        alpha=1,
        color="red",
    )
    plot.plot(
        neg_x_lin,
        neg_y_lin,
        linewidth=emph_size,
        linestyle="--",
        alpha=1,
        color="navy",
    )
    plot.set_xlim(xlim)
    plot.set_ylim(ylim)

    plot = axes[0, 2]
    stat = volatility_clustering.volatility_clustering_stats(
        log_returns=log_returns, max_lag=1000
    )
    pos_y, x_lin, alpha, beta = (
        np.mean(stat["vol_clust"], axis=1),
        stat["power_fit_x"],
        stat["alpha"],
        stat["beta"],
    )
    pos_x = np.arange(1, pos_y.size + 1)
    y_lin = np.exp(alpha) * np.power(x_lin, beta)
    plot.set(
        title="volatility clustering",
        ylabel=r"$Corr(|r_t|, |r_{t+k}|)$",
        xlabel=r"lag $k$",
        xscale="log",
        yscale="log",
    )
    plot.plot(
        pos_x,
        pos_y,
        alpha=0.8,
        marker=".",
        color="cornflowerblue",
        markersize=line_size,
        linestyle="None",
    )
    plot.plot(
        x_lin,
        y_lin,
        linestyle="--",
        linewidth=emph_size,
        color="navy",
    )

    stat = leverage_effect.leverage_effect_stats(log_returns=log_returns, max_lag=100)
    lev_eff = stat["lev_eff"]
    y = np.mean(lev_eff, axis=1)
    x = np.arange(1, y.size + 1)

    pow_c, pow_rate = stat["alpha"], stat["beta"]
    x_lin = np.linspace(np.min(x), np.max(x), num=100)
    y_pow = -np.exp(pow_c) * np.power(x_lin, pow_rate)

    plot = axes[1, 0]
    plot.set(
        title="leverage effect",
        ylabel=r"$L(k)$",
        xlabel=r"lag $k$",
        xscale="linear",
        yscale="linear",
    )
    plot.plot(
        x,
        y,
        alpha=0.8,
        marker="None",
        color="cornflowerblue",
        markersize=0,
        linestyle="-",
        linewidth=line_size,
    )
    # plot.plot(x_lin, y_lin, label="$\\tau = 50.267$", color="red", linestyle="--", alpha=1)
    plot.plot(
        x_lin[2:],
        y_pow[2:],
        color="navy",
        linestyle="--",
        linewidth=emph_size,
    )
    # plot.plot( x_lin[5:], y_lin_qiu[5:], label="Qiu paper $\\tau = 13$", color="navy", linestyle="--", alpha=0.3)
    plot.axhline(y=0, linestyle="--", c="black", alpha=0.4)

    stat = coarse_fine_volatility.coarse_fine_volatility_stats(
        log_returns=log_returns, tau=5, max_lag=100
    )
    dll, dll_x = np.mean(stat["delta_lead_lag"], axis=1), stat["delta_lead_lag_k"]
    ll, ll_x = np.mean(stat["lead_lag"], axis=1), stat["lead_lag_k"]
    argmin, alpha, beta = stat["argmin"], stat["alpha"], stat["beta"]
    plot = axes[1, 1]
    plot.set(
        title="coarse-fine volatility",
        ylabel=r"$\rho(k)$",
        xlabel=r"lag $k$",
        xscale="linear",
        yscale="linear",
    )
    plot.plot(dll_x, dll, c="blue")
    plot.plot(
        dll_x[argmin],
        dll[argmin],
        linestyle="none",
        marker=".",
        markersize=emph_size,
        c="red",
    )
    plot.plot(
        ll_x,
        ll,
        marker=".",
        color="cornflowerblue",
        markersize=line_size,
        linestyle="none",
    )
    plot.plot(
        dll_x,
        dll,
        marker="none",
        color="orange",
        markersize=0,
        linestyle="-",
        linewidth=emph_size,
    )
    x_lin = dll_x[argmin + 2 :]
    y_lin = -np.exp(alpha) * (x_lin**beta)
    plot.plot(
        x_lin,
        y_lin,
        color="red",
        alpha=1,
    )
    plot.axhline(y=0, linestyle="--", c="black", alpha=0.4)

    stat = gain_loss_asymetry.gain_loss_asymmetry_stat(
        log_price=log_price, max_lag=1000, theta=0.1
    )
    gain, loss = stat["gain"], stat["loss"]
    gain_data = np.mean(gain, axis=1)
    loss_data = np.mean(loss, axis=1)

    plot = axes[1, 2]
    plot.set(
        title="gain loss asymmetry",
        ylabel=r"return time probability",
        xlabel=r"lag $k$ in days",
        xscale="log",
        yscale="linear",
    )

    plot.plot(
        gain_data,
        alpha=0.8,
        marker=".",
        color="violet",
        markersize=line_size,
        linestyle="None",
    )
    plot.plot(
        loss_data,
        alpha=0.8,
        marker=".",
        color="cornflowerblue",
        markersize=line_size,
        linestyle="None",
    )

    max_gain, max_loss, arg_max_gain, arg_max_loss = (
        stat["max_gain"],
        stat["max_loss"],
        stat["arg_max_gain"],
        stat["arg_max_loss"],
    )

    plot.plot(
        [arg_max_gain, arg_max_gain],
        [0, max_gain],
        color="red",
        linestyle=":",
        linewidth=line_size,
    )
    plot.plot(
        [arg_max_loss, arg_max_loss],
        [0, max_loss],
        color="navy",
        linestyle=":",
        linewidth=line_size,
    )

    return fig
