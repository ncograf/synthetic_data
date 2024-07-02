import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cpuinfo
import numpy as np
import numpy.typing as npt
import pandas as pd
import real_data_loader as data
import scipy.stats
import static_stats
import stylized_score
import torch
import wandb_logging
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import SP500GanDataset
from fin_gan import FinGan
from torch.utils.data import DataLoader

import wandb


def train_fingan():
    """Fit a FinGan Neural Network. I.e. Train the network on the given data.

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
        "seq_len": 8192,
        "train_seed": 99,
        "dtype": "float32",
        "epochs": 3000,
        "batch_size": 24,
        "dist": "studentt",
        # "moment_losses" : ['mean', 'variance', 'skewness', 'kurtosis'],
        "moment_losses": [],
        "optim_gen_config": {
            "lr": 2e-4,
            "betas": (0.5, 0.999),
        },
        "optim_disc_config": {
            "lr": 1e-5,
            "betas": (0.1, 0.999),
        },
        "lr_config": {
            "gamma": 0.999,
        },
    }

    root_dir = Path(__file__).parent.parent.parent
    N_TICKS = 9216

    # setup cache for the train run
    TIME_FORMAT = "%Y_%m_%d-%H_%M_%S"
    t = datetime.now().strftime(TIME_FORMAT)
    cache = Path(f"{root_dir}/data/cache/FinGanTakahashi_{t}")
    cache.mkdir(parents=True, exist_ok=True)

    # load real data
    data_loader = data.RealDataLoader(cache=root_dir / "data/cache")
    price_data = data_loader.get_timeseries(
        "Adj Close", data_path=root_dir / "data/raw_yahoo_data"
    )

    # all colums in the dataframe must have at least seq_len non_nan elements
    non_nans = np.array(np.sum(~np.isnan(price_data), axis=0))
    price_data = price_data.drop(
        price_data.columns[non_nans <= N_TICKS], axis="columns"
    )
    price_data = price_data.iloc[-N_TICKS:]

    # accelerator is used to efficiently use resources
    set_seed(config["train_seed"])
    accelerator = Accelerator()
    device = accelerator.device

    # read available hardware
    cuda_name = None
    cpu_name = cpuinfo.get_cpu_info()["brand_raw"]
    print(f"Fin Gan Generator is using:\nCPU: {cpu_name}")
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(device)
        print(f"GPU: {cuda_name}")

    with wandb.init(tags=["FinGanTakahashi", config["dist"]] + config["moment_losses"]):
        # process data
        log_returns = np.array(price_data)
        log_returns = np.log(log_returns[1:] / log_returns[:-1])
        real_stats = stylized_score._compute_stats(log_returns, "real")

        # determine shift and scale
        data_config = {"scale": 1, "shift": 0}

        # get distribution
        real_static_stats = static_stats.static_stats(log_returns)
        mean = real_static_stats["mean"]
        std = real_static_stats["std"]
        var = real_static_stats["variance"]
        skewness = real_static_stats["skewness"]
        kurtosis = real_static_stats["kurtosis"]

        # log wandb if wandb logging is active
        if wandb.run is not None:
            wandb.config.update(config)
            wandb.config["data_stats"] = real_static_stats
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
                fit = {"loc": 0, "scale": 1}

        # read config
        gen_config = {
            "seq_len": config["seq_len"],
            "dtype": config["dtype"],
            "model": "mlp",
        }
        disc_config = {"input_dim": config["seq_len"], "dtype": config["dtype"]}
        dist_config = {"dist": config["dist"], **fit}
        batch_size = config["batch_size"]
        epochs = config["epochs"]

        # create dataset (note that the dataset will sample randomly during training (see source for more information))
        dataset = SP500GanDataset(log_returns, batch_size * 1024, config["seq_len"])
        loader = DataLoader(dataset, batch_size, pin_memory=True)

        # initialize model and optimiers
        model = FinGan(gen_config, disc_config, data_config, dist_config)
        gen_optim = torch.optim.Adam(
            model.gen.parameters(), **config["optim_gen_config"]
        )
        disc_optim = torch.optim.Adam(
            model.disc.parameters(), **config["optim_disc_config"]
        )
        bce_criterion = torch.nn.BCELoss()

        # wrap model, loader ... to get them to the right device
        loader, model, gen_optim, disc_optim = accelerator.prepare(
            loader, model, gen_optim, disc_optim
        )

        best_score = sys.float_info.max

        for epoch in range(epochs):
            gen_epoch_loss = 0
            disc_epoch_loss = 0
            epoch_time = time.time()

            # actual training
            for real_batch, y in loader:
                # read current batch size, migth not match config!
                b_size = real_batch.shape[0]

                # get data and labels
                y_real, y_fake = y[:, 0], y[:, 1]
                fake_batch: torch.Tensor = model.sample(batch_size=b_size)
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
                disc_err_real = bce_criterion(disc_y_real, y_real)
                accelerator.backward(disc_err_real)  # compute gradients

                disc_y_fake = model.disc(
                    fake_batch.detach()
                ).flatten()  # detach because generator gradients are not needed
                disc_err_fake = bce_criterion(disc_y_fake, y_fake)
                accelerator.backward(disc_err_fake)  # compute gradients

                # update discriminator
                disc_optim.step()

                #####################
                # Train generator
                #####################
                gen_optim.zero_grad()
                disc_y_fake = model.disc(
                    fake_batch
                ).flatten()  # compute with gen gradients
                gen_err = bce_criterion(disc_y_fake, y_real)

                flat_fake_batch = fake_batch.flatten()
                gen_mean = torch.mean(flat_fake_batch)
                gen_std = torch.std(flat_fake_batch)

                # Compute and add momemt losses if configured
                if "mean" in config["moment_losses"]:
                    gen_err += torch.mean(
                        torch.abs(gen_mean / (torch.abs(mean) + 1) - 1) ** 2
                    )
                if "variance" in config["moment_losses"]:
                    gen_var = torch.var(flat_fake_batch)
                    gen_err += torch.mean((gen_var / (var + 1) - 1) ** 2)
                if "skewness" in config["moment_losses"]:
                    gen_skewness = torch.mean((flat_fake_batch - gen_mean) ** 3) / (
                        gen_std**3
                    )
                    gen_err += torch.mean(
                        (torch.abs(gen_skewness) / (torch.abs(skewness) + 1) - 1) ** 2
                    )
                if "kurtosis" in config["moment_losses"]:
                    gen_kurtosis = torch.mean((flat_fake_batch - gen_mean) ** 4) / (
                        gen_std**4
                    )
                    gen_err += torch.mean((gen_kurtosis / (kurtosis + 1) - 1) ** 2)

                accelerator.backward(gen_err)

                # update generator
                gen_optim.step()

                # cumulate normalized loss
                gen_epoch_loss += gen_err.item() / len(loader)
                disc_epoch_loss += (
                    (disc_err_fake.item() + disc_err_real.item()) / len(loader) / 2
                )

            logs = {
                "gen_loss": gen_epoch_loss,
                "disc_loss": disc_epoch_loss,
                "epoch_time": time.time() - epoch_time,
                "epoch": epoch,
            }

            # get returns (note the transposition due to batch sampling)
            log_ret_sim = model.sample(batch_size=201).detach().cpu().numpy().T
            log_ret_sim = log_ret_sim * data_config["scale"] + data_config["shift"]
            old_best_score = best_score
            try:
                syn_stats = stylized_score._compute_stats(log_ret_sim, "syn")
                total_score, scores = stylized_score._stylized_score(
                    **syn_stats, **real_stats
                )
                scores["total_score"] = total_score
                best_score = np.minimum(total_score, best_score)
                logs.update(scores)
            except Exception as e:
                print(f"Expeption occured on coputing stylized statistics: {str(e)}.")
            logs["stats"] = static_stats.static_stats(log_ret_sim)

            # log eopch loss if wandb is activated
            if wandb.run is not None:
                wandb.log(logs)

            # log experiments every n epochs
            n = 100
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
                    wandb_name=f"fingan_takahashi.{epoch_name}",
                    local_path=loc_path / "model.pt",
                    desc=f"FinGAN model from the Takahashi paper after training {epoch + 1} epochs.",
                )

                wandb_logging.log_stylized_facts(
                    local_path=loc_path / "stylized_facts.png",
                    wandb_path=f"{epoch_name}/stylized_facts.png",
                    figure_title=f"Stylized Facts FinGAN Takahashi (epoch {epoch + 1})",
                    log_returns=log_ret_sim,
                )

                wandb_logging.log_temp_series(
                    local_path=loc_path / "sample_returns.png",
                    wandb_path=f"{epoch_name}/sample_returns.png",
                    figure_title=f"Simulated Log Returns FinGAN Takahashi (epoch {epoch + 1})",
                    temp_data=pd.Series(log_ret_sim[:, 0]),  # pick the first sample
                )

                wandb_logging.log_temp_series(
                    local_path=loc_path / "sample_prices.png",
                    wandb_path=f"{epoch_name}/sample_prices.png",
                    figure_title=f"Simulated Prices FinGAN Takahashi (epoch {epoch + 1})",
                    temp_data=pd.Series(
                        np.exp(np.cumsum(log_ret_sim[:, 0]))
                    ),  # pick the first sample
                )

                # print progress every n epochs
                print(
                    (
                        f"epoch: {epoch + 1:>6d}/{epochs},\tlast gen loss {gen_epoch_loss:>6.4f},\tlast disc loss {disc_epoch_loss:>6.4f},"
                    )
                )

        # after the last epoch free memory
        accelerator.free_memory()
        print("Finished training Takahashi Fin Gan!")


def sample_fingan(file: str | Path, seed: int = 99) -> npt.NDArray:
    """Generate data from the trained model


    Args:
        file (str | Path): path to the GARCH.
        seed (int, optional): manual seed. Defaults to 99.

    Returns:
        npt.NDArray: log return simulations
    """

    N_SEQ = 1000
    set_seed(seed)

    file = Path(file)

    accelerator = Accelerator()

    model_dict = torch.load(file, map_location=torch.device("cpu"))
    model = FinGan(**model_dict["init_params"])
    model.load_state_dict(model_dict["state_dict"])
    model = accelerator.prepare(model)

    # sample and bring the returns to the cpu (transpse to get dates in axis 0)
    log_returns = model.sample(batch_size=N_SEQ, unnormalize=True)
    log_returns = log_returns.detach().cpu().numpy().T

    return log_returns


if __name__ == "__main__":
    # os.environ["WANDB_MODE"] = "disabled"
    train_fingan()
