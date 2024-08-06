import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import stylized_score


def visualize_stylized_facts(
    stf: npt.ArrayLike, stf_dist: npt.ArrayLike, conf_inter: float = 0.95
) -> plt.Figure:
    """Visualizes all stilized facts and returns the plt figure

    Args:
        stf (array_like):  stylized facts
        stf_dist (array_like):  stylized facts samples from boostrap or real distribution
        conf_inter (float, optional): confidence interval. Defaults to 0.95

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

    line_size = 2
    line_color = "cornflowerblue"
    line_color2 = "violet"

    emph_size = 2
    emph_color = "navy"
    emph_color2 = "red"

    low = (1 - conf_inter) / 2
    high = 1 - (1 - conf_inter) / 2

    ###########################
    # LINEAR UNPREDICATBILITY #
    ###########################
    plot = axes[0, 0]
    lu_bstrap = stf_dist[0]
    lu_mean = stf[0]
    B = lu_bstrap.shape[1]
    interval = np.array([B * low, B * high]).astype(int)
    interval = lu_bstrap[:, interval]
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
    plot.fill_between(x, interval[:, 0], interval[:, 1], fc=line_color, alpha=0.2)

    ###############
    # HEAVY TAILS #
    ###############
    ht_bstrap = stf_dist[1]
    (
        (pos_beta, neg_beta, pos_alpha, neg_alpha),
        pos_x_fit,
        neg_x_fit,
        pos_y,
        pos_x,
        neg_y,
        neg_x,
    ) = stf[1]
    B = ht_bstrap.shape[1]
    interval = np.array([B * low, B * high]).astype(int)
    interval = ht_bstrap[:, interval]

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

    pos_x_lin = np.linspace(np.min(pos_x_fit), np.max(pos_x_fit), num=100)
    pos_y_lin = np.exp(pos_alpha) * np.power(pos_x_lin, pos_beta)
    pos_filter = (pos_y_lin > ylim[0]) & (pos_y_lin < ylim[1])
    pos_x_lin = pos_x_lin[pos_filter]
    pos_y_lin = pos_y_lin[pos_filter]

    neg_x_lin = np.linspace(np.min(neg_x_fit), np.max(neg_x_fit), num=10)
    neg_y_lin = np.exp(neg_alpha) * np.power(neg_x_lin, neg_beta)
    neg_filter = (neg_y_lin > ylim[0]) & (neg_y_lin < ylim[1])
    neg_x_lin = neg_x_lin[neg_filter]
    neg_y_lin = neg_y_lin[neg_filter]

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

    x_bstrap = (
        np.minimum(np.min(pos_x_lin), np.min(neg_x_lin)),
        np.maximum(np.max(pos_x_lin), np.max(neg_x_lin)),
    )
    x_bstrap = np.linspace(x_bstrap[0], x_bstrap[1], num=10)

    # compute positive fits
    pos_y_high = np.exp(interval[2, 0]) * np.power(x_bstrap, interval[0, 0])
    pos_y_low = np.exp(interval[2, 1]) * np.power(x_bstrap, interval[0, 1])
    plot.fill_between(x_bstrap, pos_y_high, pos_y_low, fc=emph_color2, alpha=0.1)

    # compute negative fits
    neg_y_high = np.exp(interval[3, 0]) * np.power(x_bstrap, interval[1, 0])
    neg_y_low = np.exp(interval[3, 1]) * np.power(x_bstrap, interval[1, 1])
    plot.fill_between(x_bstrap, neg_y_high, neg_y_low, fc=emph_color, alpha=0.1)
    plot.legend(loc="lower left")

    #########################
    # VOLATILITY CLUSTERING #
    #########################
    plot = axes[0, 2]
    vc_bstrap = stf_dist[2]
    vc_mean = stf[2]
    B = vc_bstrap.shape[1]
    interval = np.array([B * low, B * high]).astype(int)
    interval = vc_bstrap[:, interval]
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
    plot.fill_between(x, interval[:, 0], interval[:, 1], fc=line_color, alpha=0.2)

    ###################
    # LEVERAGE EFFECT #
    ###################
    plot = axes[1, 0]
    le_bstrap = stf_dist[3]
    le_mean = stf[3]
    B = le_bstrap.shape[1]
    interval = np.array([B * low, B * high]).astype(int)
    interval = le_bstrap[:, interval]
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
    plot.fill_between(x, interval[:, 0], interval[:, 1], fc=line_color, alpha=0.2)
    plot.axhline(y=0, linestyle="--", c="black", alpha=0.4)

    ##########################
    # COARSE FINE VOLATILITY #
    ##########################
    plot = axes[1, 1]
    cf_bstrap = stf_dist[4]
    ll_mean, ll_x, dll_mean, dll_x = stf[4]
    B = cf_bstrap.shape[1]
    interval = np.array([B * low, B * high]).astype(int)
    interval = cf_bstrap[:, interval]

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
    plot.fill_between(dll_x, interval[:, 0], interval[:, 1], fc=line_color2, alpha=0.3)
    plot.axhline(y=0, linestyle="--", c="black", alpha=0.4)

    #######################
    # GAIN LOSS ASYMMETRY #
    #######################
    plot = axes[1, 2]
    gl_bstrap = stf_dist[5]
    gl_mean = stf[5]
    n = gl_mean.size // 2
    x = np.arange(1, n + 1)
    B = gl_bstrap.shape[1]
    interval = np.array([B * low, B * high]).astype(int)
    interval = gl_bstrap[:, interval]

    plot = axes[1, 2]
    plot.set(
        title="gain loss asymmetry",
        ylabel=r"return time probability",
        xlabel=r"lag $k$ (days)",
        xscale="log",
        yscale="linear",
    )

    plot.plot(
        x,
        gl_mean[:n],
        alpha=1,
        marker=".",
        color=line_color2,
        markersize=2 * line_size,
        linestyle="None",
    )
    plot.plot(
        x,
        gl_mean[n:],
        alpha=1,
        marker=".",
        color=line_color,
        markersize=2 * line_size,
        linestyle="None",
    )
    plot.fill_between(x, interval[:n, 0], interval[:n, 1], fc=line_color2, alpha=0.2)
    plot.fill_between(x, interval[n:, 0], interval[n:, 1], fc=line_color, alpha=0.2)

    return fig


if __name__ == "__main__":
    import warnings

    import load_data
    import train_fingan
    import train_garch

    B = 100
    S = 12
    L = 2048

    real = True
    smi = True
    dax = True
    fingan = False
    garch = False

    if real:
        data = load_data.load_log_returns("sp500", min_len=9216)
        stf_dist = stylized_score.boostrap_stylized_facts(data, B, S, L)
        stf = stylized_score.compute_mean_stylized_fact(data)
        fig = visualize_stylized_facts(stf, stf_dist)
        fig.savefig("/home/nico/thesis/presentations/week22/figures/stf_sp500.png")
        plt.show()

    if smi:
        data = load_data.load_log_returns("smi", min_len=4096)
        stf_dist = stylized_score.boostrap_stylized_facts(data, B, S, L)
        stf = stylized_score.compute_mean_stylized_fact(data)
        fig = visualize_stylized_facts(stf, stf_dist)
        fig.savefig("/home/nico/thesis/presentations/week22/figures/stf_smi.png")
        plt.show()

    if dax:
        data = load_data.load_log_returns("dax", min_len=4096)
        stf_dist = stylized_score.boostrap_stylized_facts(data, B, S, L)
        stf = stylized_score.compute_mean_stylized_fact(data)
        fig = visualize_stylized_facts(stf, stf_dist)
        fig.savefig("/home/nico/thesis/presentations/week22/figures/stf_dax.png")
        plt.show()

    if fingan:
        fingan = train_fingan.load_fingan(
            "/home/nico/thesis/code/data/cache/results/epoch_43/model.pt"
        )

        def fingan_sampler(S):
            return train_fingan.sample_fingan(model=fingan, batch_size=S)

        stf_dist = stylized_score.stylied_facts_from_model(fingan_sampler, B, S)
        stf = stylized_score.compute_mean_stylized_fact(fingan_sampler(B))
        visualize_stylized_facts(stf, stf_dist)
        plt.show()

    if garch:
        garch_models = train_garch.load_garch(
            "/home/nico/thesis/code/data/cache/garch_experiments/GARCH_ged_2024_07_14-03_28_28/garch_models.pt"
        )

        def garch_sampler(S):
            with warnings.catch_warnings(action="ignore"):
                return train_garch.sample_garch(garch_models, S, L)

        stf_dist = stylized_score.stylied_facts_from_model(garch_sampler, B, S)
        stf = stylized_score.compute_mean_stylized_fact(garch_sampler(B))
        visualize_stylized_facts(stf, stf_dist)
        plt.show()
