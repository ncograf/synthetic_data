import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import click
import cpuinfo
import load_data
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats
import static_stats
import stylized_loss
import stylized_score
import torch
import wandb_logging
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import SP500DataSet
from fourier_flow import FourierFlow
from torch.utils.data import DataLoader
from type_converter import TypeConverter

import wandb


def _train_fourierflow(conf: Dict[str, Any] = {}):
    """Fit a FourierFlow Neural Network. I.e. Train the network on the given data.

    The method logs intermediate results every few epochs
    """
    # decide whether or not to log to
    if os.getenv("WANDB_MODE") not in [None, "disabled"]:
        # check settings (the envvars should be available in the poetry env (if configured correctly))
        if os.getenv("WANDB_API_KEY") is None:
            raise EnvironmentError("WANDB_API_KEY is not set!")
        if os.getenv("WANDB_ENTITY") is None:
            raise EnvironmentError("WANDB_ENTITY is not set!")
        if os.getenv("WANDB_PROJECT") is None:
            raise EnvironmentError("WANDB_PROJECT is not set!")

    # define training config and train model
    config = {
        "train_seed": 99,
        "dtype": "float32",
        "seq_len": 4096,
        "symbols": [],
        "fourier_flow_config": {
            "hidden_dim": 2048,
            "num_layer": 2,
        },
        "epochs": 500,
        "batch_size": 64,
        "num_batches": 128,
        "dist": "normal",
        # "stylized_losses": ['lu', 'le', 'cf', 'vc'],
        "stylized_losses": [],
        "stylized_lambda": 5,
        "optim_gen_config": {
            "lr": 1e-5,
        },
        "n_bootstraps": 256,
        "n_samples_per_bstrap": 8,
    }

    config.update(conf)

    N_TICKS = 9216

    # setup cache for the train run
    TIME_FORMAT = "%Y_%m_%d-%H_%M_%S"
    t = datetime.now().strftime(TIME_FORMAT)
    cache = Path(os.environ["RESULT_DIR"]) / f"FourierFlow_{t}"
    cache.mkdir(parents=True, exist_ok=True)

    # load real data
    symbols = config["symbols"]
    log_returns = load_data.load_log_returns("sp500", N_TICKS, symbols)

    # accelerator is used to efficiently use resources
    set_seed(config["train_seed"])
    accelerator = Accelerator()
    device = accelerator.device

    # read available hardware
    cuda_name = None
    cpu_name = cpuinfo.get_cpu_info()["brand_raw"]
    print(f"FourierFlow training is using:\nCPU: {cpu_name}")
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(device)
        print(f"GPU: {cuda_name}")

    with wandb.init(
        tags=["FourierFlow", config["dist"]] + config["stylized_losses"] + symbols
    ):
        # process data
        bootstrap_samples = config["n_samples_per_bstrap"]
        bootstraps = config["n_bootstraps"]
        real_stf = stylized_score.boostrap_stylized_facts(
            log_returns, bootstraps, bootstrap_samples, L=config["seq_len"]
        )

        # get distribution
        real_static_stats = static_stats.static_stats(log_returns)
        mean = real_static_stats["mean"]
        std = real_static_stats["std"]

        # determine shift and scale
        # data_config = {"scale": 1, "shift": 0}

        # log wandb if wandb logging is active
        if wandb.run is not None:
            wandb.config.update(config)
            wandb.config["train_config.cpu"] = cpu_name
            wandb.config["train_config.gpu"] = cuda_name
            wandb.config["train_config.device"] = device
            wandb.define_metric("*", step_metric="epoch")

        # set distribution
        match config["dist"]:
            case "studentt":
                df, loc, scale = scipy.stats.t.fit(log_returns.flatten())
                fit = {"df": df, "loc": loc, "scale": scale}
            case "normal":
                fit = {"loc": mean, "scale": std}
            case "cauchy":
                loc, scale = scipy.stats.cauchy.fit(log_returns.flatten())
                fit = {"loc": loc, "scale": scale}
            case "stdnormal":
                config["dist"] = "normal"
                fit = {"loc": 0, "scale": 1}
            case "laplace":
                loc, scale = scipy.stats.laplace.fit(log_returns.flatten())
                fit = {"loc": loc, "scale": scale}
        dist_config = {"dist": config["dist"], **fit}

        # read config
        fourier_flow_config = config["fourier_flow_config"]
        fourier_flow_config["seq_len"] = config["seq_len"]
        batch_size = config["batch_size"]
        num_batches = config["num_batches"]
        stl = config["stylized_lambda"]
        epochs = config["epochs"]
        dtype = TypeConverter.str_to_numpy(config["dtype"])

        # create dataset (note that the dataset will sample randomly during training (see source for more information))
        dataset = SP500DataSet(
            log_returns.astype(dtype), batch_size * num_batches, config["seq_len"]
        )
        loader = DataLoader(dataset, batch_size, pin_memory=True)

        # initialize model and optimiers
        model = FourierFlow(**fourier_flow_config, dist_config=dist_config)
        optim = torch.optim.Adam(model.parameters(), **config["optim_gen_config"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)

        # wrap model, loader ... to get them to the right device
        loader, model, optim, scheduler = accelerator.prepare(
            loader, model, optim, scheduler
        )

        best_score = sys.float_info.max

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_time = time.time()

            # actual training
            for real_batch, _ in loader:
                optim.zero_grad()

                fake_batch, log_prob_z, log_jac_det = model(real_batch)
                loss = torch.mean(-log_prob_z - log_jac_det)

                fbt = fake_batch.transpose(0, 1)
                rbt = real_batch.transpose(0, 1)
                if "lu" in conf["stylized_losses"]:
                    lu_loss = stylized_loss.lu_loss(fbt, rbt)
                    loss += stl * lu_loss
                if "le" in conf["stylized_losses"]:
                    le_loss = stylized_loss.le_loss(fbt, rbt)
                    loss += stl * le_loss
                if "cf" in conf["stylized_losses"]:
                    cf_loss = stylized_loss.cf_loss(fbt, rbt)
                    loss += stl * cf_loss
                if "vc" in conf["stylized_losses"]:
                    vc_loss = stylized_loss.vc_loss(fbt, rbt)
                    loss += stl * vc_loss

                accelerator.backward(loss)  # compute gradients

                # update model
                optim.step()

                epoch_loss += loss.item()

            # scheduler.step()

            logs = {
                "loss": epoch_loss / len(loader),
                "epoch_time": time.time() - epoch_time,
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch,
            }

            # get returns (note the transposition due to batch sampling)
            def sampler(S):
                return model.sample(batch_size=S).detach().cpu().numpy().T

            old_best_score = best_score
            try:
                syn_stf = stylized_score.stylied_facts_from_model(
                    sampler, bootstraps, bootstrap_samples
                )
                total_score, scores, _ = stylized_score.stylized_score(
                    real_stf, syn_stf
                )
                scores = {
                    "stylized_scores/lu": scores[0],
                    "stylized_scores/ht": scores[1],
                    "stylized_scores/vc": scores[2],
                    "stylized_scores/le": scores[3],
                    "stylized_scores/cf": scores[4],
                    "stylized_scores/gl": scores[5],
                }
                scores["total_score"] = total_score
                best_score = np.minimum(total_score, best_score)
                logs.update(scores)
            except Exception as e:
                print(f"Expeption occured on coputing stylized statistics: {str(e)}.")
            sampled_data = sampler(bootstraps)
            stf = stylized_score.compute_mean_stylized_fact(sampled_data)

            # log eopch loss if wandb is activated
            if wandb.run is not None:
                wandb.log(logs)

            # log experiments every n epochs
            n = epochs // 10
            if epoch % n == n - 1 or epoch == epochs - 1 or old_best_score > best_score:
                # create pseudo identifier
                epoch_name = f"epoch_{epoch + 1}" if epoch < epochs - 1 else "final"

                # comptue local path for logggin
                loc_path = Path(cache) / epoch_name
                loc_path.mkdir(parents=True, exist_ok=True)

                # read out model state (NOTE THE DICT KEYS ARE READ FROM THE SAMPLE FUNCTION DON'T CHANGE)
                model_dict = {
                    "state_dict": model.state_dict(),
                    "init_params": model.get_model_info(),
                    "network": str(model),
                    "fit_score": logs["total_score"]
                    if ("total_score" in logs)
                    else np.nan,
                }

                if epoch_name == "final":
                    with (loc_path / "model_info.json").open("w") as file:
                        json.dump(config, file)

                # log model and stats (note that the methods ALWAYS log locally)
                wandb_logging.log_model(
                    model_dict,
                    wandb_name=None,  # f"fingan_takahashi.{epoch_name}", # dont upload to wandb
                    local_path=loc_path / "model.pt",
                    desc=f"Fourier Flow after training {epoch + 1} epochs.",
                )

                wandb_logging.log_stylized_facts(
                    local_path=loc_path / "stylized_facts.png",
                    wandb_path=f"{epoch_name}/stylized_facts.png",
                    figure_title=f"Stylized Facts Fourier Flow (epoch {epoch + 1})",
                    stf=stf,
                    stf_dist=syn_stf,
                )

                wandb_logging.log_temp_series(
                    local_path=loc_path / "sample_returns.png",
                    wandb_path=f"{epoch_name}/sample_returns.png",
                    figure_title=f"Simulated Log Returns Fourier Flow (epoch {epoch + 1})",
                    temp_data=pd.Series(sampled_data[:, 0]),  # pick the first sample
                )

                wandb_logging.log_temp_series(
                    local_path=loc_path / "sample_prices.png",
                    wandb_path=f"{epoch_name}/sample_prices.png",
                    figure_title=f"Simulated Prices Fourier Flow (epoch {epoch + 1})",
                    temp_data=pd.Series(
                        np.exp(np.cumsum(sampled_data[:, 0]))
                    ),  # pick the first sample
                )

                # print progress every n epochs
                print(
                    (
                        f"epoch: {epoch + 1:>6d}/{epochs},\tlast loss {epoch_loss / len(loader):>6.4f}"
                    )
                )

        # after the last epoch free memory
        accelerator.free_memory()
        print("Finished training Fourier Flow!")


def load_fourierflow(file: str) -> FourierFlow:
    """Load model from memory

    Args:
        file (str): File to get model from

    Returns:
        FourierFlow: initialized model
    """
    file = Path(file)

    accelerator = Accelerator()

    model_dict = torch.load(file, map_location=torch.device("cpu"))
    model = FourierFlow(**model_dict["init_params"])
    model.load_state_dict(model_dict["state_dict"])
    model = accelerator.prepare(model)

    return model


def sample_fourierflow(model: FourierFlow, batch_size: int = 24) -> npt.NDArray:
    """Generate data from the trained model

    Args:
        file (FourierFlow): Initialized model
        batch_size (int): number of seqences to sample from model

    Returns:
        npt.NDArray: log return simulations
    """

    # sample and bring the returns to the cpu (transpse to get dates in axis 0)
    log_returns = model.sample(batch_size=batch_size)
    log_returns = log_returns.detach().cpu().numpy().T

    return log_returns


@click.command()
@click.option(
    "--dist",
    type=click.STRING,
    default="normal",
    help='Distribution to sample from ["normal", "studentt", "cauchy", "laplace"]',
)
@click.option(
    "--stylized-loss",
    "-s",
    multiple=True,
    default=[],
    # default=["lu", "le", "cf", "vc"],
    help='Stylized losses to use ["lu", "le", "cf", "vc"]',
)
@click.option(
    "--symbols",
    "-a",
    multiple=True,
    default=[],
    help="Symbols to be included in the training. Leave empty to include all.",
)
@click.option(
    "--learning-rate",
    "-l",
    default=1e-5,
    help="Learning rate for Adam optimizer",
)
@click.option("--stylized-lambda", "-y", default=1, help="Factor for stylized loss")
def train_fourierflow(
    dist: str,
    stylized_loss: List[str],
    symbols: List[str],
    learning_rate: float,
    stylized_lambda: float,
):
    config = {
        "dist": dist,
        "symbols": list(symbols),
        "stylized_losses": list(stylized_loss),
        "stylized_lambda": stylized_lambda,
        "optim_gen_config": {
            "lr": learning_rate,
        },
    }
    _train_fourierflow(config)


if __name__ == "__main__":
    # os.environ["WANDB_MODE"] = "disabled"
    train_fourierflow()
