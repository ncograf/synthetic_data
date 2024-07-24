import bootstrap
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


def visualize_stylized_facts(log_returns: npt.ArrayLike) -> plt.Figure:
    """Visualizes all stilized facts and returns the plt figure

    Args:
        log_returns (array_like):  (n_timesteps x m_stocks) or (n_timesteps) return data.

    Returns:
        plt.Figure: matplotlib figure ready to plot / save
    """

    # configure plt plots
    figure_style = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "figure.titlesize": 22,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "font.size": 14,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "figure.dpi": 96,
        "legend.loc": "upper right",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.1,
        "figure.constrained_layout.hspace": 0,
        "figure.constrained_layout.w_pad": 0.1,
        "figure.constrained_layout.wspace": 0,
    }
    subplot_layout = {
        "ncols": 3,
        "nrows": 2,
        "sharex": "none",
        "sharey": "none",
    }

    plt.rcParams.update(figure_style)
    # plt.style.use(["science", "ieee"])

    aspect_ratio = 10 / 16
    width = 16
    height = width * aspect_ratio
    fig, axes = plt.subplots(
        **subplot_layout, constrained_layout=True, figsize=(width, height)
    )

    # prepare data
    log_returns = np.asarray(log_returns)
    if log_returns.ndim == 1:
        log_returns = log_returns.reshape((-1, 1))
    log_price = np.cumsum(log_returns, axis=0)

    line_size = 2
    line_color = "cornflowerblue"
    line_color2 = "violet"

    emph_size = 2
    emph_color = "navy"
    emph_color2 = "red"

    B = 500
    S = 24
    L = 4096

    tail_quantile = 0.1  # used for the fit in the heavy tails

    ###########################
    # LINEAR UNPREDICATBILITY #
    ###########################
    plot = axes[0, 0]
    stf = linear_unpredictability.linear_unpredictability
    kwargs = {"max_lag": 1000}
    lu_bstrap, lu_mean = bootstrap.boostrap_distribution(
        log_returns, stf, B, S, L, **kwargs
    )
    ninty_five = np.array([B * 0.025, B * 0.975]).astype(int)
    ninty_five = lu_bstrap[:, ninty_five]
    x = np.arange(1, lu_mean.shape[0] + 1)

    plot.set(
        title="linear unpredictability",
        ylabel=r"$Corr(r_t, r_{t+k})$",
        xlabel=r"lag $k$ (days)",
        xscale="log",
        yscale="linear",
        ylim=(-0.25, 0.25),
    )
    plot.plot(
        x,
        lu_mean,
        marker=".",
        color=line_color,
        markersize=line_size,
        linestyle="None",
    )
    plot.fill_between(x, ninty_five[:, 0], ninty_five[:, 1], fc=line_color, alpha=0.2)

    ###############
    # HEAVY TAILS #
    ###############
    def stf(data):
        pd, pb, nd, nb = heavy_tails.discrete_pdf(data)
        (pb, pa), _ = heavy_tails.heavy_tails(pd, pb, tail_quantile)
        (nb, na), _ = heavy_tails.heavy_tails(nd, nb, tail_quantile)
        return np.array([pb, pa, nb, na]).reshape(-1, 1)

    pos_y, pos_x, neg_y, neg_x = heavy_tails.discrete_pdf(log_returns)

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
        alpha=0.7,
        marker=".",
        color=line_color2,
        markersize=line_size,
        linestyle="None",
    )
    plot.plot(
        neg_x,
        neg_y,
        alpha=0.7,
        marker=".",
        color=line_color,
        markersize=line_size,
        linestyle="None",
    )

    # the the plot limits for the drawing
    ylim = plot.get_ylim()
    log_lim, log_mean = np.log(ylim), np.mean(np.log(ylim))
    ylim = np.exp((log_lim - log_mean) * 0.9 + log_mean)

    # compute positive fit and cosmetics
    (pos_beta, pos_alpha), pos_x_fit = heavy_tails.heavy_tails(
        pos_y, pos_x, tq=tail_quantile
    )
    pos_x_lin = np.linspace(np.min(pos_x_fit), np.max(pos_x_fit), num=100)
    pos_y_lin = np.exp(pos_alpha) * np.power(pos_x_lin, pos_beta)
    pos_filter = (pos_y_lin > ylim[0]) & (pos_y_lin < ylim[1])
    pos_x_lin = pos_x_lin[pos_filter]
    pos_y_lin = pos_y_lin[pos_filter]

    # compute negative fit and cosmetics
    (neg_beta, neg_alpha), neg_x_fit = heavy_tails.heavy_tails(
        neg_y, neg_x, tq=tail_quantile
    )
    neg_x_lin = np.linspace(np.min(neg_x_fit), np.max(neg_x_fit), num=10)
    neg_y_lin = np.exp(neg_alpha) * np.power(neg_x_lin, neg_beta)
    neg_filter = (neg_y_lin > ylim[0]) & (neg_y_lin < ylim[1])
    neg_x_lin = neg_x_lin[neg_filter]
    neg_y_lin = neg_y_lin[neg_filter]

    ht_bstrap, _ = bootstrap.boostrap_distribution(
        log_returns, stf, B=500, S=24, L=4096
    )
    x_bstrap = (
        np.minimum(np.min(pos_x_lin), np.min(neg_x_lin)),
        np.maximum(np.max(pos_x_lin), np.max(neg_x_lin)),
    )

    # plot the fitted lines
    plot.plot(
        pos_x_lin,
        pos_y_lin,
        label=f"pos. $c_p \\, \\tilde{{r}}_t^{{\\, {pos_beta:.2f}}}$",
        linewidth=line_size,
        linestyle="--",
        alpha=1,
        color=emph_color2,
    )
    plot.plot(
        neg_x_lin,
        neg_y_lin,
        label=f"neg. $c_n \\, \\tilde{{r}}_t^{{\\, {neg_beta:.2f}}}$",
        linewidth=line_size,
        linestyle="--",
        alpha=1,
        color=emph_color,
    )

    ninty_five = np.array([B * 0.025, B * 0.975]).astype(int)
    ninty_five = ht_bstrap[:, ninty_five]
    x_bstrap = np.linspace(x_bstrap[0], x_bstrap[1], num=10)

    # compute positive fits
    pos_y_high = np.exp(ninty_five[1, 0]) * np.power(x_bstrap, ninty_five[0, 0])
    pos_y_low = np.exp(ninty_five[1, 1]) * np.power(x_bstrap, ninty_five[0, 1])
    plot.fill_between(x_bstrap, pos_y_high, pos_y_low, fc=emph_color2, alpha=0.1)

    # compute negative fits
    neg_y_high = np.exp(ninty_five[3, 0]) * np.power(x_bstrap, ninty_five[2, 0])
    neg_y_low = np.exp(ninty_five[3, 1]) * np.power(x_bstrap, ninty_five[2, 1])
    plot.fill_between(x_bstrap, neg_y_high, neg_y_low, fc=emph_color, alpha=0.1)
    plot.legend(loc="lower left")

    #########################
    # VOLATILITY CLUSTERING #
    #########################
    plot = axes[0, 2]
    stf = volatility_clustering.volatility_clustering
    kwargs = {"max_lag": 1000}
    vc_bstrap, vc_mean = bootstrap.boostrap_distribution(
        log_returns, stf, B, S, L, **kwargs
    )
    ninty_five = np.array([B * 0.025, B * 0.975]).astype(int)
    ninty_five = vc_bstrap[:, ninty_five]
    x = np.arange(1, vc_mean.shape[0] + 1)

    plot.set(
        title="volatility clustering",
        ylabel=r"$Corr(|r_t|, |r_{t+k}|)$",
        xlabel=r"lag $k$ (days)",
        xscale="log",
        yscale="log",
    )
    plot.plot(
        x,
        vc_mean,
        alpha=1,
        marker=".",
        color=line_color,
        markersize=line_size,
        linestyle="None",
    )
    plot.fill_between(x, ninty_five[:, 0], ninty_five[:, 1], fc=line_color, alpha=0.2)

    ###################
    # LEVERAGE EFFECT #
    ###################
    plot = axes[1, 0]
    stf = leverage_effect.leverage_effect
    kwargs = {"max_lag": 1000}
    le_bstrap, le_mean = bootstrap.boostrap_distribution(
        log_returns, stf, B, S, L, **kwargs
    )
    ninty_five = np.array([B * 0.025, B * 0.975]).astype(int)
    ninty_five = le_bstrap[:, ninty_five]
    x = np.arange(1, le_mean.shape[0] + 1)

    plot.set(
        title="leverage effect",
        ylabel=r"$L(k)$",
        xlabel=r"lag $k$ (days)",
        xscale="linear",
        yscale="linear",
    )
    plot.plot(
        x,
        le_mean,
        alpha=1,
        marker=".",
        color=line_color,
        markersize=line_size,
        linestyle="None",
    )
    plot.fill_between(x, ninty_five[:, 0], ninty_five[:, 1], fc=line_color, alpha=0.2)
    plot.axhline(y=0, linestyle="--", c="black", alpha=0.4)

    ##########################
    # COARSE FINE VOLATILITY #
    ##########################
    plot = axes[1, 1]
    kwargs = {"tau": 5, "max_lag": 120}

    def stf(data, tau, max_lag):
        _, _, dll, _ = coarse_fine_volatility.coarse_fine_volatility(data, tau, max_lag)
        return dll

    dll_bstrap, _ = bootstrap.boostrap_distribution(log_returns, stf, B, S, L, **kwargs)
    ll_mean, ll_x, dll_mean, dll_x = coarse_fine_volatility.coarse_fine_volatility(
        log_returns, **kwargs
    )
    ll_mean, dll_mean = np.mean(ll_mean, axis=1), np.mean(dll_mean, axis=1)
    ninty_five = np.array([B * 0.025, B * 0.975]).astype(int)
    ninty_five = dll_bstrap[:, ninty_five]

    plot.set(
        title="coarse-fine volatility",
        ylabel=r"$\rho(k)$",
        xlabel=r"lag $k$ (days)",
        xscale="linear",
        yscale="linear",
    )
    plot.plot(
        ll_x,
        ll_mean,
        marker=".",
        color=line_color,
        markersize=line_size,
        linestyle="none",
    )
    plot.plot(
        dll_x,
        dll_mean,
        marker="none",
        color=line_color2,
        markersize=0,
        linestyle="-",
        linewidth=emph_size,
    )
    plot.fill_between(
        dll_x, ninty_five[:, 0], ninty_five[:, 1], fc=line_color2, alpha=0.3
    )
    plot.axhline(y=0, linestyle="--", c="black", alpha=0.4)

    ##########################
    # COARSE FINE VOLATILITY #
    ##########################
    gain, loss = gain_loss_asymetry.gain_loss_asymmetry(
        log_price=log_price, max_lag=1000, theta=0.1
    )
    gain_mean = np.mean(gain, axis=1)
    loss_mean = np.mean(loss, axis=1)

    plot = axes[1, 2]
    plot.set(
        title="gain loss asymmetry",
        ylabel=r"return time probability",
        xlabel=r"lag $k$ (days)",
        xscale="log",
        yscale="linear",
    )

    plot.plot(
        gain_mean,
        alpha=1,
        marker=".",
        color=line_color2,
        markersize=2 * line_size,
        linestyle="None",
    )
    plot.plot(
        loss_mean,
        alpha=1,
        marker=".",
        color=line_color,
        markersize=2 * line_size,
        linestyle="None",
    )

    return fig


if __name__ == "__main__":
    data = load_data.load_log_returns("sp500")
    visualize_stylized_facts(data)
    plt.show()
