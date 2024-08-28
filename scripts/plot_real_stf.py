import json
import os
import random
from pathlib import Path

import load_data
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plot_table
import scienceplots
import stylized_score
import visualize_scores
import wasserstein_distance
from matplotlib.legend_handler import HandlerTuple

# keep this in the file
scienceplots.__name__

# Set seeds.
SEED = 12345
np.random.seed(SEED)
random.seed(SEED)

# select output
pgf = True

# laod data
min_len = 9216
price_df = load_data.load_prices("sp500")
data_info = {}
log_ret, symbols = load_data.get_log_returns(price_df, min_len, False, data_info)

# set experiment settings
out_dir = Path(os.getenv("THESIS_DIR")) / "figure"
data_dir = Path(os.getenv("DATA_DIR")) / "real_runs"
data_dir.mkdir(parents=True, exist_ok=True)
B = 500
S = 8
L = 4096
R = 15

# compare random splits
if (data_dir / "split_data.json").exists():
    with (data_dir / "split_data.json").open() as file:
        split_data = json.load(file)["data"]
else:
    scores = []
    total_scores = []
    w_scores = []
    names = []
    for _ in range(R):
        n_sym = len(symbols)
        all_idx = set(range(n_sym))
        idx_one = list(np.random.choice(list(all_idx), n_sym // 2))
        idx_two = list(all_idx.difference(idx_one))

        stf_one = stylized_score.boostrap_stylized_facts(log_ret[:, idx_one], B, S, L)
        stf_two = stylized_score.boostrap_stylized_facts(log_ret[:, idx_two], B, S, L)

        tot, sc, _ = stylized_score.stylized_score(stf_one, stf_two)

        w = [
            wasserstein_distance.compute_wasserstein_correlation(
                log_ret[:, idx_one], log_ret[:, idx_two], 1, S=40000
            )
        ]

        names.append("")
        w_scores.append(w)
        scores.append(sc)
        total_scores.append(tot)

    split_data = list(zip(total_scores, scores, w_scores, names))
    with (data_dir / "split_data.json").open("w") as file:
        json.dump({"data": split_data}, file)

# compare stocks with index
if (data_dir / "stock_data.json").exists():
    with (data_dir / "stock_data.json").open() as file:
        stock_data = json.load(file)["data"]
else:
    stock_scores = []
    w_scores = []
    stock_total_scores = []
    stf = stylized_score.boostrap_stylized_facts(log_ret, B, S, L)
    n_sym = len(symbols)
    for i in range(n_sym):
        stf_two = stylized_score.boostrap_stylized_facts(log_ret[:, [i]], B, S, L)
        tot, sc, _ = stylized_score.stylized_score(stf, stf_two)
        w = [
            wasserstein_distance.compute_wasserstein_correlation(
                log_ret, log_ret[:, [i]], 1, S=40000
            )
        ]

        w_scores.append(w)
        stock_scores.append(sc)
        stock_total_scores.append(tot)

    stock_data = list(zip(stock_total_scores, stock_scores, w_scores, symbols))
    with (data_dir / "stock_data.json").open("w") as file:
        json.dump({"data": stock_data}, file)


# compare indices amng each other
if (data_dir / "index_data.json").exists():
    with (data_dir / "index_data.json").open() as file:
        index_data = json.load(file)["data"]
else:
    smi_info = {}
    smi, _ = load_data.get_log_returns(
        load_data.load_prices("smi"), 4096, False, smi_info
    )
    dax_info = {}
    dax, _ = load_data.get_log_returns(
        load_data.load_prices("dax"), 4096, False, dax_info
    )
    L = 3072
    index_data = []
    stf_sp = stylized_score.boostrap_stylized_facts(log_ret, B, S, L)
    stf_smi = stylized_score.boostrap_stylized_facts(smi, B, S, L)

    stf_dax = stylized_score.boostrap_stylized_facts(dax, B, S, L)

    tot, sc, _ = stylized_score.stylized_score(stf_dax, stf_smi)
    w = [wasserstein_distance.compute_wasserstein_correlation(smi, dax, 1, S=40000)]
    index_data.append((tot, sc, w, "DAX -- SMI"))
    tot, sc, _ = stylized_score.stylized_score(stf_smi, stf_dax)
    index_data.append((tot, sc, w, "SMI -- DAX"))
    tot, sc, _ = stylized_score.stylized_score(stf_sp, stf_dax)
    w = [wasserstein_distance.compute_wasserstein_correlation(log_ret, dax, 1, S=40000)]
    index_data.append((tot, sc, w, "S\\&P500 -- DAX"))
    tot, sc, _ = stylized_score.stylized_score(stf_dax, stf_sp)
    index_data.append((tot, sc, w, "DAX -- S\\&P500"))
    tot, sc, _ = stylized_score.stylized_score(stf_sp, stf_smi)
    w = [wasserstein_distance.compute_wasserstein_correlation(log_ret, smi, 1, S=40000)]
    index_data.append((tot, sc, w, "S\\&P500 -- SMI"))
    tot, sc, _ = stylized_score.stylized_score(stf_smi, stf_sp)
    index_data.append((tot, sc, w, "SMI -- S\\&P500"))
    with (data_dir / "index_data.json").open("w") as file:
        json.dump({"data": index_data}, file)


if pgf:
    # configure plt plots
    figure_style = {
        "text.usetex": True,
        "pgf.preamble": r"\usepackage{amsmath}",
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
    }
    matplotlib.use("pgf")
    plt.style.use(["science", "ieee"])
else:
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
    "nrows": 1,
    "sharex": "none",
    "sharey": "none",
}
plt.rcParams.update(figure_style)


if pgf:
    scale = 1.1
    width = 5.106 * scale
    plot_scale = 0.3

else:
    plot_scale = 1
    width = 16

aspect_ratio = 10 / 16
height = width * aspect_ratio
fig, axes = plt.subplots(
    **subplot_layout,
    constrained_layout=True,
    figsize=(width, height),
)

twin_axes = [axes[i].twinx() for i in range(len(axes))]
for tax, ax in zip(twin_axes[:-1], axes[:-1]):
    ax.sharey(axes[-1])
    tax.sharey(twin_axes[-1])
visualize_scores.plot_axes(
    split_data, axes[0], twin_axes[0], "Split data scores (a)", True, scale=plot_scale
)
visualize_scores.plot_axes(
    stock_data, axes[1], twin_axes[1], "Stock scores (b)", True, scale=plot_scale
)
art_stf, art_wd, avg_lin, v_stf, v_wd = visualize_scores.plot_axes(
    index_data, axes[2], twin_axes[2], "Index scores (c)", True, scale=plot_scale
)
for tax, ax in zip(twin_axes[:-1], axes[1:]):
    tax.set_ylabel(None)
    ax.set_ylabel(None)


art = list(reversed(list(zip(avg_lin, art_stf, art_wd))))
all_ = list(
    [
        r"max $\mathcal{S}$",
        r"med $\mathcal{S}$",
        r"min $\mathcal{S}$",
        r"range ($\mathcal{S}^{\Theta}, W_1$)",
    ]
)

leg = fig.legend(
    art + [(v_stf, v_wd)],
    all_,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    ncol=4,
    handlelength=3,
    bbox_to_anchor=(0.5, 1),
    loc="lower center",
)

if pgf:
    fig.savefig(out_dir / "real_stf_data.pgf")
    plot_table.create_table(
        "split_real_stf_data",
        "real_stf_data",
        "(a)",
        "Split experiemnt data",
        plot_table.prepare_data(split_data),
        out_dir.parent / "tables",
    )
    plot_table.create_table(
        "stock_real_stf_data",
        "real_stf_data",
        "(b)",
        "Single stock experiemnt data",
        plot_table.prepare_data(stock_data),
        out_dir.parent / "tables",
    )
    plot_table.create_table(
        "index_real_stf_data",
        "real_stf_data",
        "(c)",
        "Index experiemnt data",
        plot_table.prepare_data(index_data),
        out_dir.parent / "tables",
    )
else:
    plt.savefig(out_dir / "real_stf_data.png")
