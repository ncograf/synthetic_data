import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import real_data_loader
import stylized_score
import train_garch

seed = 3599
N_STOCKS = 9216

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
        "FIGARCH",
        {
            "model": "FIGARCH",
            "max": 5,
            "par": ["p", "q"],
            "power": 2,
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
real_loader = real_data_loader.RealDataLoader(cache=data_dir / "cache")
real_data = real_loader.get_timeseries(
    col_name="Adj Close", data_path=data_dir / "raw_yahoo_data", update_all=False
)
real_data = real_data.drop(["SPY"], axis="columns")
nan_mask = ~np.isnan(real_data)  # returns pd.dataframe
num_non_nans: pd.DataFrame = np.sum(nan_mask, axis=0)
stocks = num_non_nans.iloc[num_non_nans.values >= N_STOCKS].index
first_date = real_data.index[-N_STOCKS]
real_data = real_data.loc[first_date:, stocks]
real_data = real_data.loc[:, np.all(~np.isnan(real_data), axis=0)]

np_data = np.asarray(real_data)
real_log_ret = np.log(np_data[1:] / np_data[:-1])
real_log_ret[np.abs(real_log_ret) >= 2] = 0  # clean data

path = Path(os.environ["RESULT_DIR"]) / "garch_experiments.json"
experiments = []

N = 1

for model, pg in grid:
    config = {"vol": pg["model"], "power": pg["power"], "mean": "Constant"}
    for dist in pg["dist"]:
        for p in range(1, pg["max"] + 1):
            par_dict = {s: p for s in pg["par"]}
            config = config.copy()
            config.update({"dist": dist})
            config.update(par_dict)

            # try:
            dir = train_garch._train_garch(config)

            total_score = []
            individual_socore = []
            try:
                for i in range(N):
                    log_returns = train_garch.sample_garch(dir, seed=(seed + i))
                    tot, ind = stylized_score.stylized_score(real_log_ret, log_returns)
                    total_score.append(tot)
                    individual_socore.append(list(ind.values()))

                experiments.append(
                    {
                        "score": np.nanmean(total_score),
                        "score_std": np.nanstd(total_score),
                        "ind_scores": list(np.nanmean(individual_socore, axis=0)),
                        "ind_scores_std": list(np.nanstd(individual_socore, axis=0)),
                        "config": config,
                    }
                )
            except:  # noqa E722
                experiments.append({"score": 2**31, "config": config})

            experiments.sort(key=lambda x: x["score"])
            with path.open("w") as file:
                file.write(json.dumps(experiments, ensure_ascii=True, indent=4))
