import numpy as np


def plot_axes(plot_data, plot, plot_, title, alt=False, scale=1):
    plot_data = sorted(plot_data, key=lambda x: x[0])
    plot_data = list(reversed(list(reversed(plot_data))))

    stf_mins = np.min(list(zip(*plot_data))[1], axis=0)
    stf_maxs = np.max(list(zip(*plot_data))[1], axis=0)

    wd_mins = np.min(list(zip(*plot_data))[2], axis=0)
    wd_maxs = np.max(list(zip(*plot_data))[2], axis=0)

    min = plot_data[0]
    med = plot_data[len(plot_data) // 2]
    max = plot_data[-1]

    line_size = 4 * scale
    marker_size = 10 * scale
    if alt:
        line_mark = ":"
        marker = "."
    else:
        line_mark = "--"
        marker = "+"
    avg_line = 2 * scale

    colors = ["#EE82EE", "#B741B7", "#7F007F"]
    alt_colors = ["#6495EC", "#324BF6", "#0000FF"]

    plot.set(
        title=title,
        ylabel=r"Stylized Score $\mathcal{S}^{\theta}$",
        xlabel=r"Stylized Fact $\theta$",
    )
    plot.minorticks_off()
    plot_.set(ylabel=r"Wasserstein distance $10^3 \times W_1(R, R')$", yscale="symlog")
    plot_.minorticks_off()
    plot.set_xticks(
        range(7),
        labels=[
            r"Linear unpredictability",
            r"Heavy tails",
            r"Volatility clustering",
            r"Leverage effect",
            r"Coarse-fine volatility",
            r"Gain-loss asymmetry",
            r"Wasserstein Distance",
        ],
        rotation=90,
    )

    plot.grid(visible=True, alpha=0.2, color=colors[1], linewidth=0.5 * scale)
    # plot_.grid(visible=True, alpha=0.2, color=alt_colors[1], linewidth=0.5)

    avg_lines = []
    for i, a in enumerate((min[0], med[0], max[0])):
        avg_lines.append(
            plot.hlines(
                a,
                0,
                5,
                color=colors[i],
                linestyle=line_mark,
                linewidth=avg_line,
                alpha=0.8,
            )
        )

    num_plot = len(min[1])
    num_plot_ = len(min[2])

    vline_stf = plot.vlines(
        np.arange(stf_mins.size),
        stf_mins,
        stf_maxs,
        color=colors[1],
        linewidth=line_size,
        alpha=0.2,
        capstyle="round",
    )
    vline_wd = plot_.vlines(
        num_plot + np.arange(wd_mins.size),
        wd_mins,
        wd_maxs,
        color=alt_colors[1],
        linewidth=line_size,
        alpha=0.2,
        capstyle="round",
    )

    artists_wd = []
    artists_stf = []

    (art,) = plot.plot(
        min[1],
        marker=marker,
        color=colors[0],
        markersize=marker_size,
        linestyle="None",
    )
    (art_,) = plot_.plot(
        num_plot + np.arange(num_plot_),
        min[2],
        marker=marker,
        color=alt_colors[0],
        markersize=marker_size,
        linestyle="None",
    )
    artists_stf.append(art)
    artists_wd.append(art_)

    (art,) = plot.plot(
        med[1],
        marker=marker,
        color=colors[1],
        markersize=marker_size,
        linestyle="None",
    )
    (art_,) = plot_.plot(
        num_plot + np.arange(num_plot_),
        med[2],
        marker=marker,
        color=alt_colors[1],
        markersize=marker_size,
        linestyle="None",
    )
    artists_stf.append(art)
    artists_wd.append(art_)

    (art,) = plot.plot(
        max[1],
        marker=marker,
        color=colors[2],
        markersize=marker_size,
        linestyle="None",
    )
    (art_,) = plot_.plot(
        num_plot + np.arange(num_plot_),
        max[2],
        marker=marker,
        color=alt_colors[2],
        markersize=marker_size,
        linestyle="None",
    )
    artists_stf.append(art)
    artists_wd.append(art_)

    return (artists_stf, artists_wd, avg_lines, vline_stf, vline_wd)
