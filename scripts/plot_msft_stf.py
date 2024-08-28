import json
import os
import random
import warnings
from pathlib import Path

import load_data
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plot_table
import scienceplots
import stylized_score
import train_fingan
import visualize_scores
import wasserstein_distance
from matplotlib.legend_handler import HandlerTuple

# keep this in the file
scienceplots.__name__

# Set seeds.
SEED = 12345
np.random.seed(SEED)
random.seed(SEED)

sym = "msft"
pgf = True

# laod data
min_len = 9216
price_df = load_data.load_prices("sp500")
data_info = {}
log_ret, symbols = load_data.get_log_returns(price_df, min_len, False, data_info)
msft_idx = symbols.index(sym.upper())
msft_ret = log_ret[:, [msft_idx]]

# set experiment settings
out_dir = Path(os.getenv("THESIS_DIR")) / "figure"
data_dir = Path(os.getenv("DATA_DIR")) / f"{sym}_fingan_runs"
data_dir.mkdir(parents=True, exist_ok=True)
B = 500
S = 8
L = 4096

# MSFT Real VS MSFT Model
if (data_dir / "rvsm.json").exists():
    with (data_dir / "rvsm.json").open() as file:
        exp_file = json.load(file)
        rvsm_exp = exp_file["exp"]
        rvsm_exp_ = exp_file["exp_"]
else:
    # laod data
    stf = stylized_score.boostrap_stylized_facts(msft_ret, B, S, L)

    exp_path = data_dir / "rvsm.json"
    rvsm_exp_ = []
    rvsm_exp = []

    for i, path in enumerate(data_dir.iterdir()):
        if (path / "final/model.pt").exists():
            print(f"Working on {str(path)}, nr {i}")

            model = train_fingan.load_fingan(path / "final/model.pt")
            with (path / "final/model_info.json").open() as file:
                info = json.load(file)

            def sampler(S):
                with warnings.catch_warnings(action="ignore"):
                    return train_fingan.sample_fingan(model, S)

            m_stf = stylized_score.stylied_facts_from_model(sampler, B, S)
            tot, ind, _ = stylized_score.stylized_score(stf, m_stf)

            w = [
                wasserstein_distance.compute_wasserstein_correlation(
                    msft_ret, sampler(20), 1, S=40000
                )
            ]

            if len(info["stylized_losses"]) == 0:
                rvsm_exp_.append((tot, ind, w, f'~{info["dist"]}'))
            else:
                rvsm_exp.append((tot, ind, w, f'~{info["dist"]} + stylized loss'))

    with exp_path.open("w") as file:
        json.dump({"exp": rvsm_exp, "exp_": rvsm_exp_}, file)

rvsm_exp.sort(key=lambda x: x[0])
rvsm_exp_.sort(key=lambda x: x[0])

# compare stocks with index
if (data_dir / "msft_data.json").exists():
    with (data_dir / "msft_data.json").open() as file:
        msft_data = json.load(file)["data"]
else:
    stf = stylized_score.boostrap_stylized_facts(log_ret, B, S, L)
    stf_two = stylized_score.boostrap_stylized_facts(msft_ret, B, S, L)
    tot, sc, _ = stylized_score.stylized_score(stf, stf_two)
    w = [
        wasserstein_distance.compute_wasserstein_correlation(
            log_ret, msft_ret, 1, S=40000
        )
    ]

    msft_data = [(tot, sc, w, "MSFT")]
    with (data_dir / "msft_data.json").open("w") as file:
        json.dump({"data": msft_data}, file)


#
if (data_dir / "model_data.json").exists():
    with (data_dir / "model_data.json").open() as file:
        exp_file = json.load(file)
        model_exp = exp_file["exp"]
        model_exp_ = exp_file["exp_"]
else:
    # laod data
    stf = stylized_score.boostrap_stylized_facts(log_ret, B, S, L)

    exp_path = data_dir / "model_data.json"
    model_exp_ = []
    model_exp = []

    for i, path in enumerate(data_dir.iterdir()):
        if (path / "final/model.pt").exists():
            print(f"Working on {str(path)}, nr {i}")

            model = train_fingan.load_fingan(path / "final/model.pt")
            with (path / "final/model_info.json").open() as file:
                info = json.load(file)

            def sampler(S):
                with warnings.catch_warnings(action="ignore"):
                    return train_fingan.sample_fingan(model, S)

            m_stf = stylized_score.stylied_facts_from_model(sampler, B, S)
            tot, ind, _ = stylized_score.stylized_score(stf, m_stf)

            w = [
                wasserstein_distance.compute_wasserstein_correlation(
                    log_ret, sampler(20), 1, S=40000
                )
            ]

            if len(info["stylized_losses"]) == 0:
                model_exp_.append((tot, ind, w, f'~{info["dist"]}'))
            else:
                model_exp.append((tot, ind, w, f'~{info["dist"]} + stylized loss'))

    with exp_path.open("w") as file:
        json.dump({"exp": model_exp, "exp_": rvsm_exp_}, file)

model_exp.sort(key=lambda x: x[0])
model_exp_.sort(key=lambda x: x[0])


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
    rvsm_exp,
    axes[0],
    twin_axes[0],
    f"Real vs Model {sym.upper()} (a)",
    scale=plot_scale,
)
visualize_scores.plot_axes(
    rvsm_exp_,
    axes[0],
    twin_axes[0],
    f"Real vs Model {sym.upper()} (a)",
    True,
    scale=plot_scale,
)
art_stf_, art_wd_, avg_lin_, _, _ = visualize_scores.plot_axes(
    model_exp_,
    axes[1],
    twin_axes[1],
    f"S\\&P 500 vs Model {sym.upper()} (b)",
    True,
    scale=plot_scale,
)
art_stf, art_wd, avg_lin, v_stf, v_wd = visualize_scores.plot_axes(
    model_exp,
    axes[1],
    twin_axes[1],
    f"S\\&P 500 vs Model {sym.upper()} (b)",
    scale=plot_scale,
)
visualize_scores.plot_axes(
    msft_data,
    axes[2],
    twin_axes[2],
    f"S\\&P 500 vs Real {sym.upper()} (c)",
    scale=plot_scale,
)

for tax, ax in zip(twin_axes[:-1], axes[1:]):
    tax.set_ylabel(None)
    ax.set_ylabel(None)

art = list(reversed(list(zip(avg_lin, art_stf, art_wd))))
art_ = list(reversed(list(zip(avg_lin_, art_stf_, art_wd_))))
all_ = [
    r"model with max $\mathcal{S}$",
    r"model ($\mathcal{S}$-loss) with max $\mathcal{S}$",
    r"range of scores ($\mathcal{S}, W_1$)",
    r"model with med $\mathcal{S}$",
    r"model ($\mathcal{S}$-loss) with med $\mathcal{S}$",
    r"model with min $\mathcal{S}$",
    r"model ($\mathcal{S}^{\Theta}$-loss) with min $\mathcal{S}$",
]

all_art = [art_[0], art[0], (v_stf, v_wd), art_[1], art[1], art_[2], art[2]]

leg = fig.legend(
    all_art,
    all_,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    ncol=3,
    handlelength=4,
    bbox_to_anchor=(0.5, 1),
    loc="lower center",
)

if pgf:
    fig.savefig(out_dir / f"{sym}_stf_data.pgf")
    plot_table.create_table(
        f"rvsm_{sym}_stf_data",
        f"{sym}_stf_data",
        "(a)",
        "Real vs Model experiemnt data",
        plot_table.prepare_data(rvsm_exp, rvsm_exp_),
        out_dir.parent / "tables",
    )
    plot_table.create_table(
        f"model_{sym}_stf_data",
        f"{sym}_stf_data",
        "(b)",
        "S\\&P 500 vs Model experiemnt data",
        plot_table.prepare_data(model_exp, model_exp_),
        out_dir.parent / "tables",
    )
    plot_table.create_table(
        f"msft_{sym}_stf_data",
        f"{sym}_stf_data",
        "(c)",
        "S\\&P 500 vs Real experiemnt data",
        plot_table.prepare_data(msft_data),
        out_dir.parent / "tables",
    )
else:
    plt.savefig(out_dir / f"{sym}_stf_data.png")
    plt.show()
