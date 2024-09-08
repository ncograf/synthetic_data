import os
import random
from pathlib import Path

import click
import load_data
import matplotlib.pyplot as plt
import numpy as np
import plot_table
import scienceplots
import stylized_score
import visualize_stylized_facts_paper
import wasserstein_distance

# keep this in the file
scienceplots.__name__


@click.command()
@click.option(
    "-p", "--png", is_flag=True, help="If flag is given the script creates a png image"
)
def main(png):
    """Create the qualitative plots and tables for gaussian noise"""

    # Set seeds.
    SEED = 12345
    np.random.seed(SEED)
    random.seed(SEED)

    pgf = not png

    # laod data
    min_len = 9216
    price_df = load_data.load_prices("sp500")
    data_info = {}
    log_ret, symbols = load_data.get_log_returns(price_df, min_len, False, data_info)

    # set experiment settings
    out_dir = Path(os.getenv("THESIS_DIR")) / "figure"
    out_dir.mkdir(parents=True, exist_ok=True)

    B = 500
    S = 8
    L = 4096
    R = 15

    # compare random splits
    stf = stylized_score.boostrap_stylized_facts(log_ret, B, S, L)
    mu = np.mean(log_ret)
    sigma = np.std(log_ret)

    def sampler(S):
        return np.random.normal(mu, sigma, (L, S))

    scores = []
    for _ in range(R):
        m_stf = stylized_score.stylied_facts_from_model(sampler, B, S)
        tot, ind, _ = stylized_score.stylized_score(stf, m_stf)

        w = [
            wasserstein_distance.compute_wasserstein_correlation(
                log_ret, sampler(20), 1, S=40000
            )
        ]

        scores.append((tot, ind, w, ""))

    stf_dist = stylized_score.stylied_facts_from_model(sampler, B, S)
    stf_m = stylized_score.compute_mean_stylized_fact(sampler(B))
    fig = visualize_stylized_facts_paper.visualize_stylized_facts_paper(
        stf_m, stf_dist, textwidth=1.15 * 5.106
    )

    if pgf:
        fig.savefig(out_dir / "gaussian_noise.pgf")
        plot_table.create_table(
            "gaussian_noise",
            "gaussian_noise",
            "",
            "Random noise experiments",
            plot_table.prepare_data(scores),
            out_dir.parent / "tables",
        )
    else:
        plt.savefig(out_dir / "gaussian_noise.png")


if __name__ == "__main__":
    main()
