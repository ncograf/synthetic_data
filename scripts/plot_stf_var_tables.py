import os
import random
from pathlib import Path

import click
import load_data
import numpy as np
import plot_table
import stylized_score


@click.command()
@click.option(
    "-e",
    "--exp",
    type=str,
    multiple=True,
    help="Takes all expriemnts to create tables for (accepts multiple values).",
)
def main(exp):
    """Create varinace statistics for the indicated experiment / data.
    Stores the results as table in the ${THESIS_DIR}/tables directory

    Options are

    - real
    - msft
    """

    # Set seeds.
    SEED = 12345
    np.random.seed(SEED)
    random.seed(SEED)

    log_ret = load_data.load_log_returns("sp500", 9216)

    B = 500
    S = 8
    L = 4096

    real = "real" in exp
    msft = "msft" in exp

    out_dir = Path(os.getenv("THESIS_DIR")) / "tables"

    if real:
        stf_dist = stylized_score.boostrap_stylized_facts(log_ret, B, S, L)
        stvars = stylized_score.stylized_variance(stf_dist)
        text = "Standard deviations of the S\&P 500 stylized facts estimators."
        table = plot_table.std_table(text, stvars, "S\&P 500", "stf_vars")
        with (out_dir / "stf_vars.tex").open("w") as file:
            file.write(table)

    if msft:
        data = load_data.load_log_returns("sp500", min_len=9216, symbols=["MSFT"])
        stf_dist = stylized_score.boostrap_stylized_facts(data, B, S, L)
        stvars = stylized_score.stylized_variance(stf_dist)
        text = "Standard deviations of the MSFT stylized facts estimators."
        table = plot_table.std_table(text, stvars, "MSFT", "stf_msft_vars")
        with (out_dir / "stf_msft_vars.tex").open("w") as file:
            file.write(table)


if __name__ == "__main__":
    main()
