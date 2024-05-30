from datetime import date
from pathlib import Path
from typing import Any, Dict

import cpuinfo
import numpy as np
import pandas as pd
import time_gan_generator
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

import wandb
import wandb.errors


class TimeGanIndexGenerator:
    def __init__(self):
        self.model_desc = "Time Gan Base Model"
        self.model_name = "time_gan_base"

    def fit(
        self,
        price_data: pd.DataFrame,
        train_config: Dict[str, any],
        cache: str | Path,
    ):
        """Fit data for the given input data

        Args:
            price_data (pd.DataFrame): Stock marked price data
            config (Dict[str, Any]): configuration for training run:
                use_cuda: bool
                train_seed
                time_gan_config:
                    hidden_dim: int
                    num_layer: int
                    embed_dim: int
                    n_stock: int
                seq_len: int
                dtype: str
                epochs: int
                batch_size: int
                lag: int
                gamma (float): Factor for exponential LRScheduler (in every epoch the new learning rate is lr * gamma)
                optim_config: config for adam optimizer (e.g. lr : float)
                lr_config : config for exponential lr_scheduler (e.g. gamma : float)
            cache (str | Path): cache path
        """
        set_seed(train_config["train_seed"])

        # check out cpu the main process is running on
        device = torch.device("cpu")
        cpu_name = cpuinfo.get_cpu_info()["brand_raw"]
        print("Fourier Flow Generator is using:")
        print(f"CPU : {cpu_name}")

        # initialize accelerator tool
        accelerator = Accelerator()
        device = accelerator.device

        cuda_name = None
        if train_config["use_cuda"] and torch.cuda.is_available():
            cuda_name = torch.cuda.get_device_name(device)
            print(f"GPU: {cuda_name}")

        symbols = list(price_data.columns)

        if wandb.run is not None:
            wandb.config["train_config.cpu"] = cpu_name
            wandb.config["train_config.gpu"] = cuda_name
            wandb.config["train_config.device"] = device

            wandb.define_metric("*", step_metric="epoch")

        # accumulate metadata for fit artifact
        metadata = {
            "model_dict": {},
            "model_set": set(),
            "model_desc": self.model_desc,
            "model_name": self.model_name,
        }

        gen = time_gan_generator.TimeGanGenerator()
        log_likelyhood = []

        for sym in symbols:
            data = price_data.loc[:, sym]
            model_dict = gen.fit(
                price_data=data, config=train_config, accelerator=accelerator, sym=sym
            )
            model_dict["symbol"] = sym
            metadata["model_dict"][sym] = f"{sym}.pt"
            metadata["model_set"].add(f"{sym}.pt")
            log_likelyhood.append(model_dict["fit_score"])

            torch.save(model_dict, cache / f"{sym}.pt")

        metadata["fit_scores"] = log_likelyhood

        accelerator.free_memory()
        return metadata

    def sample(
        self,
        sample_config: Dict[str, Any],
        metadata: Dict[str, any],
        cache: str | Path,
    ) -> pd.DataFrame:
        """Generates synthetic data for the whole index

        Args:
            sample_config (Dict[str, Any]): Configuration for:
                n_sample : number of samples to be sampled
                n_burn : number of samples to burn before sampling
                sample_seed : seed to sample from
            metadata (Dict[str, any]): Dictonary describing the filenames of the model(s)
            cache (str | Path): Path to the stored generators / models

        Raises:
            ValueError: data must be a DataFrame

        Returns:
            pd.DataFrame : DataFrame containing all the columns
        """
        set_seed(sample_config["sample_seed"])

        cache = Path(cache)

        n_burn = sample_config["n_burn"]
        n_sample = sample_config["n_sample"]

        # read metadata
        model_dict = metadata["model_dict"]

        symbols = list(model_dict.keys())

        generator = time_gan_generator.TimeGanGenerator()

        # take time index starting from today
        time_idx = pd.date_range(str(date.today()), periods=n_sample + 1, freq="D")
        generated_prices = pd.DataFrame(
            np.zeros((n_sample + 1, len(symbols))), columns=symbols, index=time_idx
        )
        generated_returns = pd.DataFrame(
            np.zeros((n_sample, len(symbols))), columns=symbols, index=time_idx[1:]
        )

        for sym in symbols:
            model_config = torch.load(cache / model_dict[sym])
            price_simulation, return_simulation = generator.sample(
                n_sample, burn=n_burn, config=model_config
            )
            generated_prices.loc[:, sym] = price_simulation
            generated_returns.loc[:, sym] = return_simulation
        return generated_prices, generated_returns
