import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import real_data_loader
import stylized_score
import train_garch
from sklearn.model_selection import ParameterGrid

seed = 3599
N_STOCKS = 9216

grid = [
    (
        "GARCH",
        {
            "p": [1, 2],
            "o": [0, 1, 2],
            "q": [1, 2],
            "power": [0.5, 1, 2, 3],
            "dist": ["normal", "t", "skewt", "ged"],
        },
    ),
    (
        "GARCH",
        {
            "p": [3, 4],
            "o": [0, 2, 4],
            "q": [3, 4],
            "power": [0.5, 1, 2, 3],
            "dist": ["normal", "t", "skewt", "ged"],
        },
    ),
    (
        "FIGARCH",
        {
            "p": [1, 2, 3],
            "q": [0, 1, 2, 3],
            "power": [0.5, 1, 2, 3],
            "dist": ["normal", "t", "skewt", "ged"],
        },
    ),
    (
        "EGARCH",
        {
            "p": [1, 2],
            "o": [0, 1, 2],
            "q": [1, 2],
            "power": [0.5, 1, 2, 3],
            "dist": ["normal", "t", "skewt", "ged"],
        },
    ),
    (
        "EGARCH",
        {
            "p": [3, 4],
            "o": [0, 2, 4],
            "q": [3, 4],
            "power": [0.5, 1, 2, 3],
            "dist": ["normal", "t", "skewt", "ged"],
        },
    ),
    ("ARCH", {"p": [1, 2, 3, 4], "dist": ["normal", "t", "skewt", "ged"]}),
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

experiments = []

for vol, pg in grid:
    param_grid = list(ParameterGrid(pg))
    config = {"vol": vol, "mean": "Constant"}
    for grid_conf in param_grid:
        config.update(grid_conf)
        dir = train_garch._train_garch(config)
        log_returns = train_garch.sample_garch(dir, seed=seed)
        total_score, individual_scores = stylized_score.stylized_score(
            real_log_ret, log_returns
        )
        experiments.append((total_score, individual_scores, config))

experiments.sort(key=lambda x: x[0])

path = Path(os.environ["RESULT_DIR"]) / "garch_experiments.json"

with path.open("w") as file:
    file.write(json.dumps(experiments, ensure_ascii=True))
