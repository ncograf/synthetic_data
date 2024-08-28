import json
import os
import warnings
from pathlib import Path

import load_data
import numpy as np
import stylized_score
import train_garch
import wasserstein_distance

seed = 3599
N_STOCKS = 9216
B = 256
S = 8

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

# get data
data_dir = Path(os.environ["DATA_DIR"])

log_ret = load_data.load_log_returns("sp500")
log_ret[np.abs(log_ret) >= 2] = 0  # clean data
stf = stylized_score.boostrap_stylized_facts(log_ret, B, S, 8192)

path = Path(os.environ["RESULT_DIR"]) / "garch_experiments.json"
experiments = []
if path.exists():
    with path.open() as file:
        experiments = json.load(file)

N = 50

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

            total_score = []
            individual_socore = []
            try:
                model = train_garch.load_garch(dir / "model.pt")

                def sampler(S):
                    with warnings.catch_warnings(action="ignore"):
                        return train_garch.sample_garch(model, S)

                m_stf = stylized_score.stylied_facts_from_model(sampler, B, S)
                tot, ind, _ = stylized_score.stylized_score(stf, m_stf)
                total_score.append(tot)
                individual_socore.append(list(ind.values()))

                w = [
                    wasserstein_distance.compute_wasserstein_correlation(
                        log_ret,
                        sampler(300),
                        1,
                        S=40000,  # 300 will sample from most fits
                    )
                ]

                experiments.append(
                    {
                        "score": np.nanmean(total_score),
                        "wd_score": np.nanmean(w),
                        "score_std": np.nanstd(total_score),
                        "ind_scores": list(np.nanmean(individual_socore, axis=0)),
                        "ind_scores_std": list(np.nanstd(individual_socore, axis=0)),
                        "path": str(dir),
                        "config": config,
                    }
                )
            except:  # noqa E722
                experiments.append({"score": 2**31, "config": config})

            experiments.sort(key=lambda x: x["score"])
            with path.open("w") as file:
                file.write(json.dumps(experiments, ensure_ascii=True, indent=4))
