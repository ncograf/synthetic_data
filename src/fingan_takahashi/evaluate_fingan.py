import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import real_data_loader
import stylized_score
import torch
import train_fingan

N_STOCKS = 9216
SEED = 10032

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

dir = Path("/home/nico/thesis/fingan_experiments")
experiments = []

for model_dir in dir.iterdir():
    if model_dir.is_dir():
        print()
        print()
        print(model_dir.stem)

        for epoch_dir in model_dir.iterdir():
            reg_match = re.search(r"^epoch_(\d+)|final", epoch_dir.stem)

            config = {
                "model": "fingan",
                "epoch": reg_match[1] if reg_match[1] is not None else reg_match[0],
            }
            model_dict = torch.load(
                epoch_dir / "model.pt", map_location=torch.device("cpu")
            )
            config.update(model_dict["init_params"])

            log_returns = train_fingan.sample_fingan(epoch_dir / "model.pt", seed=SEED)
            total_score, individual_scores = stylized_score.stylized_score(
                real_log_ret, log_returns
            )
            experiments.append((total_score, individual_scores, config))


experiments.sort(key=lambda x: x[0])

path = Path(os.environ["RESULT_DIR"]) / "garch_experiments.json"

with path.open("w") as file:
    file.write(json.dumps(experiments, ensure_ascii=True))
