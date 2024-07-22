import json
import os
import re
from pathlib import Path

import load_data
import stylized_score
import torch
import train_fingan

N_STOCKS = 9216
SEED = 10032

real_log_ret = load_data.load_log_returns("sp500", N_STOCKS)
dir = Path("/home/nico/thesis/code/data/test")
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

path = Path(os.environ["RESULT_DIR"]) / "fingan_experiments.json"

with path.open("w") as file:
    file.write(json.dumps(experiments, ensure_ascii=True))
