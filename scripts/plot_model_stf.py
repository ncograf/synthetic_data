import json
import os
import random
import time
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
import train_fourier_flow
import train_garch
import train_real_nvp
import visualize_scores
import wasserstein_distance
from matplotlib.legend_handler import HandlerTuple

# keep this in the file
scienceplots.__name__


def compute_garch_scores(run_dir: Path, B, S, L):
    exp_path = run_dir / "exp.json"
    if exp_path.exists():
        with (run_dir / "exp.json").open() as file:
            exp_file = json.load(file)
            exp = exp_file["exp"]
    else:
        # laod data
        min_len = 9216
        log_ret = load_data.load_log_returns("sp500", min_len)
        stf = stylized_score.boostrap_stylized_facts(log_ret, B, S, L)

        exp = []

        for i, path in enumerate(run_dir.iterdir()):
            start = time.time()
            if (path / "garch_models.pt").exists():
                print(f"Working on {str(path)}, nr {i}", end="")

                model = train_garch.load_garch(path / "garch_models.pt")
                with (path / "meta_data.json").open() as file:
                    info = json.load(file)
                config = info["config"]

                name = config["vol"]
                p = config["p"]
                dist = config["dist"]

                try:

                    def sampler(S):
                        with warnings.catch_warnings(action="ignore"):
                            return train_garch.sample_garch(model, S, L)

                    m_stf = stylized_score.stylied_facts_from_model(sampler, B, S)
                    tot, ind, _ = stylized_score.stylized_score(stf, m_stf)

                    w = [
                        wasserstein_distance.compute_wasserstein_correlation(
                            log_ret, sampler(20), 1, S=40000
                        )
                    ]
                except:  # noqa E722
                    pass

                exp.append((tot, ind, w, f"{name}({p}) ~ {dist}"))

                print(f" took {time.time() - start} seconds")

            with exp_path.open("w") as file:
                json.dump({"exp": exp}, file)

    exp.sort(key=lambda x: x[0])

    return exp


def compute_scores(run_dir: Path, model_load, model_sample, B, S, L):
    if (run_dir / "exp.json").exists():
        with (run_dir / "exp.json").open() as file:
            exp_file = json.load(file)
            exp = exp_file["exp"]
            exp_ = exp_file["exp_"]
    else:
        # laod data
        min_len = 9216
        log_ret = load_data.load_log_returns("sp500", min_len)
        stf = stylized_score.boostrap_stylized_facts(log_ret, B, S, L)

        exp_path = run_dir / "exp.json"
        exp_ = []
        exp = []

        for i, path in enumerate(run_dir.iterdir()):
            if (path / "final/model.pt").exists():
                print(f"Working on {str(path)}, nr {i}")

                model = model_load(path / "final/model.pt")
                with (path / "final/model_info.json").open() as file:
                    info = json.load(file)

                def sampler(S):
                    with warnings.catch_warnings(action="ignore"):
                        return model_sample(model, S)

                m_stf = stylized_score.stylied_facts_from_model(sampler, B, S)
                tot, ind, _ = stylized_score.stylized_score(stf, m_stf)

                w = [
                    wasserstein_distance.compute_wasserstein_correlation(
                        log_ret, sampler(20), 1, S=40000
                    )
                ]

                if len(info["stylized_losses"]) == 0:
                    exp_.append((tot, ind, w, f'~{info["dist"]}'))
                else:
                    exp.append((tot, ind, w, f'~{info["dist"]} + stylized loss'))

        with exp_path.open("w") as file:
            json.dump({"exp": exp, "exp_": exp_}, file)

    exp.sort(key=lambda x: x[0])
    exp_.sort(key=lambda x: x[0])

    return exp, exp_


# Set seeds.
SEED = 12345
np.random.seed(SEED)
random.seed(SEED)

# set pgf or png
pgf = True

# set experiment settings
out_dir = Path(os.getenv("THESIS_DIR")) / "figure"
data_dir = Path(os.getenv("DATA_DIR"))
B = 500
S = 8
L = 4096

# compute garch
run_dir = data_dir / "garch_runs"
garch_exp = compute_garch_scores(run_dir, B, S, L)

# fourier flow
run_dir = data_dir / "fourierflow_runs"
ff_exp, ff_exp_ = compute_scores(
    run_dir,
    train_fourier_flow.load_fourierflow,
    train_fourier_flow.sample_fourierflow,
    B,
    S,
    L,
)

# real nvw
run_dir = data_dir / "realnvp_runs"
realnvp_exp, realnvp_exp_ = compute_scores(
    run_dir, train_real_nvp.laod_real_nvp, train_real_nvp.sample_real_nvp, B, S, L
)

# fingan
run_dir = data_dir / "fingan_runs"
fingan_exp, fingan_exp_ = compute_scores(
    run_dir, train_fingan.load_fingan, train_fingan.sample_fingan, B, S, L
)

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
        "axes.titlesize": 17,
        "axes.labelsize": 13,
        "font.size": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 96,
        "legend.loc": "upper right",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.1,
        "figure.constrained_layout.hspace": 0,
        "figure.constrained_layout.w_pad": 0.1,
        "figure.constrained_layout.wspace": 0,
    }
subplot_layout = {
    "ncols": 4,
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
    garch_exp, axes[0], twin_axes[0], "GARCH (a)", True, scale=plot_scale
)
visualize_scores.plot_axes(
    fingan_exp, axes[1], twin_axes[1], "FinGAN (b)", scale=plot_scale
)
visualize_scores.plot_axes(
    fingan_exp_, axes[1], twin_axes[1], "FinGAN (b)", True, scale=plot_scale
)
visualize_scores.plot_axes(
    ff_exp, axes[2], twin_axes[2], "Fourier Flow (c)", scale=plot_scale
)
visualize_scores.plot_axes(
    ff_exp_, axes[2], twin_axes[2], "Fourier Flow (c)", True, scale=plot_scale
)
art_stf_, art_wd_, avg_lin_, _, _ = visualize_scores.plot_axes(
    realnvp_exp_, axes[3], twin_axes[3], "Real NVP (d)", True, scale=plot_scale
)
art_stf, art_wd, avg_lin, v_stf, v_wd = visualize_scores.plot_axes(
    realnvp_exp, axes[3], twin_axes[3], "Real NVP (d)", scale=plot_scale
)

for tax, ax in zip(twin_axes[:-1], axes[1:]):
    tax.set_ylabel(None)
    ax.set_ylabel(None)

art = list(reversed(list(zip(avg_lin, art_stf, art_wd))))
art_ = list(reversed(list(zip(avg_lin_, art_stf_, art_wd_))))
all_ = [
    r"max $\mathcal{S}$",
    r"max $\mathcal{S}$ with ($\mathcal{S}$-loss)",
    r"dist over experiments ($\mathcal{S}, W_1$)",
    r"med $\mathcal{S}$",
    r"med $\mathcal{S}$ with ($\mathcal{S}$-loss)",
    r"min $\mathcal{S}$",
    r"min $\mathcal{S}$ with ($\mathcal{S}$-loss)",
]

all_art = [art_[0], art[0], (v_stf, v_wd), art_[1], art[1], art_[2], art[2]]

leg = fig.legend(
    all_art,
    all_,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    ncol=3,
    handlelength=3,
    bbox_to_anchor=(0.5, 1),
    loc="lower center",
)

if pgf:
    fig.savefig(out_dir / "model_stf_data.pgf")
    # create tables
    plot_table.create_table(
        "garch_model_stf_data",
        "model_stf_data",
        "(a)",
        "GARCH experiemnt data",
        plot_table.prepare_data(garch_exp),
        out_dir.parent / "tables",
    )
    plot_table.create_table(
        "fingan_model_stf_data",
        "model_stf_data",
        "(b)",
        "FinGAN experiemnt data",
        plot_table.prepare_data(fingan_exp, fingan_exp_),
        out_dir.parent / "tables",
    )
    plot_table.create_table(
        "fourierflow_model_stf_data",
        "model_stf_data",
        "(c)",
        "Fourier Flow experiemnt data",
        plot_table.prepare_data(ff_exp, ff_exp_),
        out_dir.parent / "tables",
    )
    plot_table.create_table(
        "realnvp_model_stf_data",
        "model_stf_data",
        "(d)",
        "Real NVP experiemnt data",
        plot_table.prepare_data(realnvp_exp, realnvp_exp_),
        out_dir.parent / "tables",
    )

else:
    plt.savefig(out_dir / "model_stf_data.png")
    plt.show()
