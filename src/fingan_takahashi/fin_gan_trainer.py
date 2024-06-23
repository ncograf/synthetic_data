import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, Tuple

import cpuinfo
import numpy as np
import pandas as pd
import torch
import wandb_logging
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import SP500GanDataset
from fin_gan import FinGan
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import wandb


class FinGanTrainer:
    def fit(
        self,
        price_data: pd.DataFrame,
        config: Dict[str, Any],
        cache: str | Path,
    ):
        """Fit a FinGan Neural Network. I.e. Train the network on the given data.

        The method logs intermediate results every few epochs

        Args:
            price_data (pd.DataFrame): Stock marked price data
            config (Dict[str, Any]): configuration for training run:
                seq_len: int
                train_seed: int
                dtype: str
                epochs: int
                batch_size: int
                optim_gen_config: config for adam optimizer (e.g. lr : float)
                optim_disc_config: config for adam optimizer (e.g. lr : float)
                lr_config : config for exponential lr_scheduler (e.g. gamma : float)
            cache (str | Path): path to store the checkpoints and results

        Returns:
            Dict[str, Any]: Model description ready for sampling
                state_dict : torch model weights
                init_params : model init params
                network : network name
        """

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

        # log wandb if wandb logging is active
        if wandb.run is not None:
            wandb.config.update(config)
            wandb.config["train_config.cpu"] = cpu_name
            wandb.config["train_config.gpu"] = cuda_name
            wandb.config["train_config.device"] = device
            wandb.define_metric("*", step_metric="epoch")

        # read config
        gen_config = {"seq_len": config["seq_len"], "dtype": config["dtype"]}
        disc_config = {"input_dim": config["seq_len"], "dtype": config["dtype"]}
        batch_size = config["batch_size"]
        epochs = config["epochs"]
        dtype = config["dtype"]
        seq_len = config["seq_len"]
        opt_gen_conf = config["optim_gen_config"]
        opt_disc_conf = config["optim_disc_config"]

        # create dataset (note that the dataset will sample randomly during training (see source for more information))
        dataset = SP500GanDataset(price_data, batch_size * 1024, seq_len)
        data_scale = dataset.scale
        data_shift = dataset.shift
        loader = DataLoader(dataset, batch_size, pin_memory=True)

        # initialize model and optimiers
        model = FinGan(gen_config, disc_config, dtype, data_scale, data_shift)
        gen_optim = torch.optim.Adam(model.gen.parameters(), **opt_gen_conf)
        disc_optim = torch.optim.Adam(model.disc.parameters(), **opt_disc_conf)
        cross_entropy_loss = CrossEntropyLoss()

        # wrap model, loader ... to get them to the right device
        loader, model, gen_optim, disc_optim = accelerator.prepare(
            loader, model, gen_optim, disc_optim
        )

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_time = time.time()

            # actual training
            for real_series, y in loader:
                gen_optim.zero_grad()
                disc_optim.zero_grad()

                # read current batch size, migth not match config!
                b_size = real_series.shape[0]
                gen_series: torch.Tensor = model.sample(batch_size=b_size)
                real_series = torch.nan_to_num(real_series).to(gen_series.dtype)

                # prep data for discriminator and dicriminate
                all_series = torch.unsqueeze(
                    torch.cat([real_series, gen_series], dim=0), dim=1
                )
                disc_y = model.disc(all_series)

                # y will contain labels for real and synthetic data already
                y = torch.cat([y[:, 0], y[:, 1]], dim=0)
                loss = cross_entropy_loss(disc_y.flatten(), y)

                # update model
                accelerator.backward(loss)
                gen_optim.step()
                disc_optim.step()

                # cumulate normalized loss
                epoch_loss += loss.item() / len(loader)

            # log eopch loss if wandb is activated
            if wandb.run is not None:
                wandb.log(
                    {
                        "loss": epoch_loss,
                        "epoch_time": time.time() - epoch_time,
                        "epoch": epoch,
                    }
                )

            # log experiments every n epochs
            n = 1
            if epoch % n == n - 1 or epoch == epochs - 1:
                # create pseudo identifier
                epoch_name = f"epoch_{epoch + 1}" if epoch < epochs - 1 else "final"

                # comptue local path for logggin
                loc_path = Path(cache) / epoch_name
                loc_path.mkdir(parents=True, exist_ok=True)

                # get prices and returns (note the transposition due to batch sampling)
                log_ret_sim = gen_series.detach().cpu().numpy().T
                log_ret_sim = log_ret_sim * data_scale + data_shift

                # read out model state (NOTE THE DICT KEYS ARE READ FROM THE SAMPLE FUNCTION DON'T CHANGE)
                model_dict = {
                    "state_dict": model.state_dict(),
                    "init_params": model.get_model_info(),
                    "network": str(model),
                    "fit_score": epoch_loss,
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
                    figure_title=f"Simulated Log Returns FinGAN Takahashi (epoch {epoch + 1}, flattend 24 samples)",
                    temp_data=pd.Series(log_ret_sim[:, 0]),  # pick the first sample
                )

                wandb_logging.log_temp_series(
                    local_path=loc_path / "sample_prices.png",
                    wandb_path=f"{epoch_name}/sample_prices.png",
                    figure_title=f"Simulated Prices FinGAN Takahashi (epoch {epoch + 1}, flattened 24 samples)",
                    temp_data=pd.Series(
                        np.exp(np.cumsum(log_ret_sim[:, 0]))
                    ),  # pick the first sample
                )

                # print progress every n epochs
                print(
                    (f"epoch: {epoch + 1:>8d}/{epochs},\tlast loss {epoch_loss:>8.4f},")
                )

        # after the last epoch free memory
        accelerator.free_memory()
        print("Finished training Takahashi Fin Gan!")

    def sample(
        self, model_dict: Dict[str, any], seed: int, n_samples: int
    ) -> Tuple[pd.DataFrame]:
        """Generate data from the trained model

        Args:
            model_dict (Dict[str, Any]): model information
                state_dict : torch model weights
                init_params : model init params
            seed (int): seed to be set
            n_samples (int): number of samples to create

        Returns:
            Tuple[pd.DataFrame]: return data (seq_len x n_samples)
        """
        set_seed(seed)
        accelerator = Accelerator()

        model = FinGan(**model_dict["init_params"])
        model.load_state_dict(model_dict["state_dict"])
        model = accelerator.prepare(model)

        # sample and bring the returns to the cpu (transpse to get dates in axis 0)
        log_returns = model.sample(n_samples, unnormalize=True)
        log_returns = log_returns.detach().cpu().numpy().T

        # generate pandas dataframes
        time_idx = pd.date_range(
            str(date.today()), periods=log_returns.shape[0], freq="D"
        )
        pd_log_returns = pd.DataFrame(log_returns, index=time_idx)

        return pd_log_returns
