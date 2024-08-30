import json
import os
import random
from pathlib import Path

import click
import numpy as np
import train_garch


@click.command()
def main():
    """Train garch models for the predefined grid"""

    # Set seeds.
    SEED = 12345
    np.random.seed(SEED)
    random.seed(SEED)

    grid = [
        (
            "ARCH",
            {
                "model": "ARCH",
                "max": 5,
                "par": ["p"],
                "power": 2,
                "dist": ["normal", "t", "skewt", "ged"],
            },
        ),
        (
            "GARCH",
            {
                "model": "GARCH",
                "max": 5,
                "par": ["p", "q"],
                "power": 2,
                "dist": ["normal", "t", "skewt", "ged"],
            },
        ),
        (
            "GJR-GARCH",
            {
                "model": "GARCH",
                "max": 5,
                "par": ["p", "o", "q"],
                "power": 2,
                "dist": ["normal", "t", "skewt", "ged"],
            },
        ),
        (
            "TARCH",
            {
                "model": "GARCH",
                "max": 5,
                "par": ["p", "o", "q"],
                "power": 1,
                "dist": ["normal", "t", "skewt", "ged"],
            },
        ),
        (
            "EGARCH",
            {
                "model": "EGARCH",
                "max": 5,
                "par": ["p", "q"],
                "power": 2,
                "dist": ["normal", "t", "skewt", "ged"],
            },
        ),
    ]

    path = Path(os.environ["RESULT_DIR"]) / "garch_runs/garch_experiments.json"
    path.parent.mkdir(exist_ok=True, parents=True)
    experiments = []
    if path.exists():
        with path.open() as file:
            experiments = json.load(file)

    for model, pg in grid:
        config = {"vol": pg["model"], "power": pg["power"], "mean": "Constant"}
        for dist in pg["dist"]:
            par_grid = [{s: p for s in pg["par"]} for p in range(1, pg["max"] + 1)]

            for par_dict in par_grid:
                config = config.copy()
                config.update({"dist": dist})
                config.update(par_dict)

                exists = False
                for exp in experiments:
                    if exp["config"] == config:
                        exists = True
                        break

                if exists:
                    continue

                # try:
                dir = train_garch._train_garch(config)

                try:
                    experiments.append(
                        {"path": str(dir), "config": config, "model": model}
                    )
                except:  # noqa E722
                    experiments.append({"config": config})

                with path.open("w") as file:
                    file.write(json.dumps(experiments, ensure_ascii=True, indent=4))


if __name__ == "__main__":
    main()
