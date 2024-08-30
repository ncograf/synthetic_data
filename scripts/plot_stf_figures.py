import json
import os
import random
import warnings
from pathlib import Path

import click
import load_data
import matplotlib.pyplot as plt
import numpy as np
import stylized_score
import train_fingan
import train_fourier_flow
import train_garch
import train_real_nvp
import visualize_stylized_facts_paper


@click.command()
@click.option(
    "-e",
    "--exp",
    type=str,
    multiple=True,
    help="Takes all expriemnts to create figures for (accepts multiple values).",
)
def main(exp):
    """Create qualitative plots of stylized facts for the indicated experiment / data.
    Stores the results as table in the ${THESIS_DIR}/figure directory

    Moreover the caller needs to ensure that the necessary expriments exsits
    and are available in ${DATA_DIR}.

    Options are:

    - real
    - msft
    - fingan
    - msft_fingan
    - fourierflow
    - realnvp
    - garch
    - smi
    - dax
    - nvr
    """

    # Set seeds.
    SEED = 12345
    np.random.seed(SEED)
    random.seed(SEED)

    log_ret = load_data.load_log_returns("sp500", 9216)

    B = 500
    S = 8
    L = 4096

    data_dir = Path(os.getenv("DATA_DIR"))
    out_dir = Path(os.getenv("THESIS_DIR")) / "figure"

    if "real" in exp:
        stf_dist = stylized_score.boostrap_stylized_facts(log_ret, B, S, L)
        stf = stylized_score.compute_mean_stylized_fact(log_ret)
        fig = visualize_stylized_facts_paper.visualize_stylized_facts_paper(
            stf, stf_dist, textwidth=1.15 * 5.106
        )
        plt.savefig(out_dir / "st_fact.pgf")

    if "smi" in exp:
        data = load_data.load_log_returns("smi", 4096)
        stf_dist = stylized_score.boostrap_stylized_facts(data, B, S, 3072)
        stf = stylized_score.compute_mean_stylized_fact(data)
        fig = visualize_stylized_facts_paper.visualize_stylized_facts_paper(
            stf, stf_dist, textwidth=1.15 * 5.106
        )
        plt.savefig(out_dir / "stf_smi.pgf")

    if "dax" in exp:
        data = load_data.load_log_returns("dax", 4096)
        stf_dist = stylized_score.boostrap_stylized_facts(data, B, S, 3072)
        stf = stylized_score.compute_mean_stylized_fact(data)
        fig = visualize_stylized_facts_paper.visualize_stylized_facts_paper(
            stf, stf_dist, textwidth=1.15 * 5.106
        )
        plt.savefig(out_dir / "stf_dax.pgf")

    if "mstf" in exp:
        data = load_data.load_log_returns("sp500", min_len=9216, symbols=["MSFT"])
        stf_dist = stylized_score.boostrap_stylized_facts(data, B, S, L)
        stf = stylized_score.compute_mean_stylized_fact(data)
        fig = visualize_stylized_facts_paper.visualize_stylized_facts_paper(
            stf, stf_dist, textwidth=1.15 * 5.106
        )
        fig.savefig(out_dir / "stf_mstf.pgf")

    if "nvr" in exp:
        data = load_data.load_log_returns("sp500", min_len=9216, symbols=["NVR"])
        stf_dist = stylized_score.boostrap_stylized_facts(data, B, S, L)
        stf = stylized_score.compute_mean_stylized_fact(data)
        fig = visualize_stylized_facts_paper.visualize_stylized_facts_paper(
            stf, stf_dist, textwidth=1.15 * 5.106
        )
        fig.savefig(out_dir / "stf_nvr.pgf")

    if "fingan" in exp:
        dist = "studentt"
        stylized_loss_len = 4
        for i, path in enumerate((data_dir / "fingan_runs").iterdir()):
            if (path / "final/model.pt").exists():
                model = train_fingan.load_fingan(path / "final/model.pt")
                with (path / "final/model_info.json").open() as file:
                    info = json.load(file)

                if not (
                    info["dist"] == dist
                    and stylized_loss_len == len(info["stylized_losses"])
                ):
                    continue

                def sampler(S):
                    with warnings.catch_warnings(action="ignore"):
                        return train_fingan.sample_fingan(model, S)

                stf_dist = stylized_score.stylied_facts_from_model(sampler, B, S)
                stf = stylized_score.compute_mean_stylized_fact(sampler(B))
                fig = visualize_stylized_facts_paper.visualize_stylized_facts_paper(
                    stf, stf_dist, textwidth=1.15 * 5.106
                )
                fig.savefig(out_dir / "stf_fingan.pgf")
                break

    if "msft_fingan" in exp:
        dist = "normal"
        stylized_loss_len = 4
        for i, path in enumerate((data_dir / "msft_fingan_runs").iterdir()):
            if (path / "final/model.pt").exists():
                model = train_fingan.load_fingan(path / "final/model.pt")
                with (path / "final/model_info.json").open() as file:
                    info = json.load(file)

                if not (
                    info["dist"] == dist
                    and stylized_loss_len == len(info["stylized_losses"])
                ):
                    continue

                def sampler(S):
                    with warnings.catch_warnings(action="ignore"):
                        return train_fingan.sample_fingan(model, S)

                stf_dist = stylized_score.stylied_facts_from_model(sampler, B, S)
                stf = stylized_score.compute_mean_stylized_fact(sampler(B))
                fig = visualize_stylized_facts_paper.visualize_stylized_facts_paper(
                    stf, stf_dist, textwidth=1.15 * 5.106
                )
                fig.savefig(out_dir / "stf_msft_fingan.pgf")
                break

    if "fourierflow" in exp:
        dist = "laplace"
        stylized_loss_len = 4
        for i, path in enumerate((data_dir / "fourierflow_runs").iterdir()):
            if (path / "final/model.pt").exists():
                model = train_fourier_flow.load_fourierflow(path / "final/model.pt")
                with (path / "final/model_info.json").open() as file:
                    info = json.load(file)

                if not (
                    info["dist"] == dist
                    and stylized_loss_len == len(info["stylized_losses"])
                ):
                    continue

                def sampler(S):
                    with warnings.catch_warnings(action="ignore"):
                        return train_fourier_flow.sample_fourierflow(model, S)

                stf_dist = stylized_score.stylied_facts_from_model(sampler, B, S)
                stf = stylized_score.compute_mean_stylized_fact(sampler(B))
                fig = visualize_stylized_facts_paper.visualize_stylized_facts_paper(
                    stf, stf_dist, textwidth=1.15 * 5.106
                )
                fig.savefig(out_dir / "stf_fourierflow.pgf")
                break

    if "realnvp" in exp:
        dist = "laplace"
        stylized_loss_len = 0
        for i, path in enumerate((data_dir / "realnvp_runs").iterdir()):
            if (path / "final/model.pt").exists():
                model = train_real_nvp.laod_real_nvp(path / "final/model.pt")
                with (path / "final/model_info.json").open() as file:
                    info = json.load(file)

                if not (
                    info["dist"] == dist
                    and stylized_loss_len == len(info["stylized_losses"])
                ):
                    continue

                def sampler(S):
                    with warnings.catch_warnings(action="ignore"):
                        return train_real_nvp.sample_real_nvp(model, S)

                stf_dist = stylized_score.stylied_facts_from_model(sampler, B, S)
                stf = stylized_score.compute_mean_stylized_fact(sampler(B))
                fig = visualize_stylized_facts_paper.visualize_stylized_facts_paper(
                    stf, stf_dist, textwidth=1.15 * 5.106
                )
                fig.savefig(out_dir / "stf_realnvp.pgf")
                break

    if "garch" in exp:
        dist = "normal"
        p = 3
        for path in (data_dir / "garch_runs").iterdir():
            if (path / "garch_models.pt").exists():
                model = train_garch.load_garch(path / "garch_models.pt")
                with (path / "meta_data.json").open() as file:
                    info = json.load(file)
                info = info["config"]

                if not (info["dist"] == dist and p == info["p"]):
                    continue

                try:

                    def sampler(S):
                        with warnings.catch_warnings(action="ignore"):
                            return train_garch.sample_garch(model, S, L)

                    stf_dist = stylized_score.stylied_facts_from_model(sampler, B, S)
                    stf = stylized_score.compute_mean_stylized_fact(sampler(B))
                    fig = visualize_stylized_facts_paper.visualize_stylized_facts_paper(
                        stf, stf_dist, textwidth=1.15 * 5.106
                    )
                    fig.savefig(out_dir / "stf_garch.pgf")
                    break

                except:  # noqa E722
                    pass
