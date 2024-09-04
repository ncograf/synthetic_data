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
import stylized_score
import torch
import wandb_logging
from accelerate import Accelerator
from accelerate.utils import set_seed
from c_flow import CFlow
from datasets import SP500DataSet
from torch.utils.data import DataLoader
from type_converter import TypeConverter

import wandb


def _train_cflow(conf: Dict[str, Any] = {}):
    """Fit a RealNVP Neural Network. I.e. Train the network on the given data.

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
        "seq_len": 1024,
        "train_seed": 99,
        "dtype": "float32",
        "epochs": 200,
        "batch_size": 128,
        "num_batches": 512,
        "dist": "normal",
        "symbols": [],
        "stylized_losses": [],
        "stylized_lambda": 5,
        "c_flow_config": {
            "hidden_dim": 48,
            "preview": 2,
            "n_layer": 4,
        },
        # "stylized_losses": ['lu', 'le', 'cf', 'vc'],
        "optim_gen_config": {
            "lr": 1e-4,
            # "betas": (0.5, 0.999),
        },
        "optim_disc_config": {
            "lr": 1e-5,
            "betas": (0.1, 0.999),
        },
        # "losses" : ["gan", "ml"],
        "losses": ["ml"],
        "n_bootstraps": 24,
        "n_samples_per_bstrap": 8,
    }

    config.update(conf)

    N_TICKS = 9216

    # setup cache for the train run
    TIME_FORMAT = "%Y_%m_%d-%H_%M_%S"
    t = datetime.now().strftime(TIME_FORMAT)
    cache = Path(os.environ["RESULT_DIR"]) / f"CFlow_{t}"
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
    print(f"CFlow training is using:\nCPU: {cpu_name}")
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(device)
        print(f"GPU: {cuda_name}")

    with wandb.init(tags=["CFlow"] + config["stylized_losses"] + symbols):
        # process data
        bootstrap_samples = config["n_samples_per_bstrap"]
        bootstraps = config["n_bootstraps"]
        real_stf = stylized_score.boostrap_stylized_facts(
            log_returns, bootstraps, bootstrap_samples, L=config["seq_len"]
        )

        # log wandb if wandb logging is active
        if wandb.run is not None:
            wandb.config.update(config)
            wandb.config["train_config.cpu"] = cpu_name
            wandb.config["train_config.gpu"] = cuda_name
            wandb.config["train_config.device"] = device
            wandb.define_metric("*", step_metric="epoch")

        # read config
        c_flow_config = config["c_flow_config"]
        seq_len = config["seq_len"]
        c_flow_config["disc_seq_len"] = seq_len
        batch_size = config["batch_size"]
        num_batches = config["num_batches"]
        # stl = config["stylized_lambda"]
        epochs = config["epochs"]
        dtype = TypeConverter.str_to_numpy(config["dtype"])

        # create dataset (note that the dataset will sample randomly during training (see source for more information))
        dataset = SP500DataSet(
            log_returns.astype(dtype),
            batch_size * num_batches,
            config["seq_len"],
        )
        loader = DataLoader(dataset, batch_size, pin_memory=True)

        # initialize model and optimiers
        model = CFlow(**c_flow_config)
        # model.set_normilizing(log_returns)
        optim = torch.optim.Adam(model.gen.parameters(), **config["optim_gen_config"])
        disc_optim = torch.optim.Adam(
            model.disc.parameters(), **config["optim_disc_config"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)

        # wrap model, loader ... to get them to the right device
        loader, model, optim, disc_optim, scheduler = accelerator.prepare(
            loader, model, optim, disc_optim, scheduler
        )

        bce_criterion = torch.nn.BCELoss()
        best_score = sys.float_info.max

        for epoch in range(epochs):
            ml_loss = 0
            gen_loss = 0
            disc_loss = 0
            epoch_time = time.time()

            # actual training
            for real_batch, y in loader:
                if "gan" in config["losses"]:
                    # get data and labels
                    b_size = real_batch.shape[0]
                    y_real, y_fake = y[:, 0], y[:, 1]
                    fake_batch: torch.Tensor = model.sample(
                        n=b_size, seq_len=seq_len, n_burn=2
                    )
                    fake_batch = torch.reshape(
                        fake_batch, (b_size, 1, -1)
                    )  # add channel dimension
                    real_batch = torch.nan_to_num(real_batch).to(fake_batch.dtype)
                    real_batch = torch.reshape(
                        real_batch, (b_size, 1, -1)
                    )  # add channel dimension

                    #######################
                    # Train discriminator
                    #######################
                    disc_optim.zero_grad()
                    disc_y_real = model.disc(real_batch).flatten()
                    y_real = y_real.to(disc_y_real.dtype)
                    disc_err_real = bce_criterion(disc_y_real, y_real)
                    accelerator.backward(disc_err_real)  # compute gradients

                    disc_y_fake = model.disc(
                        fake_batch.detach()
                    ).flatten()  # detach because generator gradients are not needed
                    y_fake = y_fake.to(disc_y_fake.dtype)
                    disc_err_fake = bce_criterion(disc_y_fake, y_fake)
                    accelerator.backward(disc_err_fake)  # compute gradients

                    # update discriminator
                    disc_optim.step()

                    #####################
                    # Train generator
                    #####################
                    optim.zero_grad()
                    disc_y_fake = model.disc(
                        fake_batch
                    ).flatten()  # compute with gen gradients
                    gen_err = bce_criterion(disc_y_fake, y_real)
                    accelerator.backward(gen_err)  # compute gradients
                    optim.step()

                    gen_loss += gen_err.item()
                    disc_loss += (disc_err_fake.item() + disc_err_real.item()) * 0.5

                elif "ml" in config["losses"]:
                    _, log_prob_z, log_jac_det = model(real_batch)

                    loss = torch.mean(-log_prob_z - log_jac_det)
                    accelerator.backward(loss)  # compute gradients

                    # update model
                    optim.step()

                    ml_loss += loss.item()

            # scheduler.step()

            logs = {
                "gen_loss": gen_loss / len(loader),
                "disc_loss": disc_loss / len(loader),
                "loss": ml_loss / len(loader),
                "epoch_time": time.time() - epoch_time,
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch,
            }

            # get returns (note the transposition due to batch sampling)
            def sampler(S):
                # sample batch
                return model.sample(S, seq_len).detach().cpu().numpy().T

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

                # log model and stats (note that the methods ALWAYS log locally)
                wandb_logging.log_model(
                    model_dict,
                    wandb_name=None,  # f"fingan_takahashi.{epoch_name}", # dont upload to wandb
                    local_path=loc_path / "model.pt",
                    desc=f"CFlow after training {epoch + 1} epochs.",
                )

                wandb_logging.log_stylized_facts(
                    local_path=loc_path / "stylized_facts.png",
                    wandb_path=f"{epoch_name}/stylized_facts.png",
                    figure_title=f"Stylized Facts CFlow (epoch {epoch + 1})",
                    stf=stf,
                    stf_dist=syn_stf,
                )

                wandb_logging.log_temp_series(
                    local_path=loc_path / "sample_returns.png",
                    wandb_path=f"{epoch_name}/sample_returns.png",
                    figure_title=f"Simulated Log Returns CFlow (epoch {epoch + 1})",
                    temp_data=pd.Series(sampled_data[:, 0]),  # pick the first sample
                )

                wandb_logging.log_temp_series(
                    local_path=loc_path / "sample_prices.png",
                    wandb_path=f"{epoch_name}/sample_prices.png",
                    figure_title=f"Simulated Prices CFlow (epoch {epoch + 1})",
                    temp_data=pd.Series(
                        np.exp(np.cumsum(sampled_data[:, 0]))
                    ),  # pick the first sample
                )

                # print progress every n epochs
                print(
                    (
                        f"epoch: {epoch + 1:>6d}/{epochs},\tepoch loss {ml_loss / len(loader):>6.4f}"
                    )
                )

        # after the last epoch free memory
        accelerator.free_memory()
        print("Finished training CFlow!")


def laod_c_flow(file: str) -> CFlow:
    """Load model from memory

    Args:
        file (str): File to get model from

    Returns:
        CFlow: initialized model
    """
    file = Path(file)

    accelerator = Accelerator()

    model_dict = torch.load(file, map_location=torch.device("cpu"))
    model = CFlow(**model_dict["init_params"])
    model.load_state_dict(model_dict["state_dict"])
    model = accelerator.prepare(model)

    return model


def sample_c_flow(
    model: CFlow, batch_size: int = 24, seq_len: int = 2048
) -> npt.NDArray:
    """Generate data from the trained model

    Args:
        file (CFlow): Initialized model
        batch_size (int): number of seqences to sample from model

    Returns:
        npt.NDArray: log return simulations
    """

    # sample and bring the returns to the cpu (transpse to get dates in axis 0)
    log_returns = model.sample(n=batch_size, seq_len=seq_len)
    log_returns = log_returns.detach().cpu().numpy().T

    return log_returns


@click.command()
@click.option(
    "--losses",
    "-s",
    multiple=True,
    default=["ml"],
    # default=["st", "gan", "ml"]
    help="Which losses should be used for training",
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
    default=1e-4,
    help="Learning rate for Adam optimizer",
)
@click.option(
    "--seq-len",
    "-L",
    default=1024,
    help="Sequence lenght used for sampling and training",
)
@click.option(
    "--epochs",
    "-e",
    default=200,
    help="Number of epochs",
)
def train_cflow(
    losses: List[str],
    symbols: List[str],
    learning_rate: float,
    seq_len: int,
    epochs: int,
):
    config = {
        "symbols": list(symbols),
        "losses": list(losses),
        "seq_len": seq_len,
        "epochs": epochs,
        "optim_gen_config": {
            "lr": learning_rate,
        },
    }
    _train_cflow(config)
    _train_cflow()


if __name__ == "__main__":
    # os.environ["WANDB_MODE"] = "disabled"
    train_cflow()
