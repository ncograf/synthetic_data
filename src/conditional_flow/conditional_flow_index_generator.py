from datetime import date
from pathlib import Path
from typing import Any, Dict, Tuple
from warnings import warn

import base_index_generator
import conditional_flow_generator
import cpuinfo
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

import wandb
import wandb.errors


class ConditionalFlowIndexGenerator(base_index_generator.BaseIndexGenerator):
    def __init__(self):
        self.model_name = "cond_flow"
        self.model_desc = "Conditional Flow Model"

    def fit(
        self, price_data: pd.DataFrame, train_config: Dict[str, any], cache: str | Path
    ) -> Dict[str, Any]:
        """Fit data for the given input data

        Args:
            price_data (pd.DataFrame): Stock marked price datsa
            config(Dict[str, Any]): configuration for training run:
                use_cuda: bool
                train_seed: int
                cond_flow_config:
                    hidden_dim (int): dimension of the hidden layers needs to be even
                    dim (int): dimension of the output / input (equals to the number of stocks to predict)
                    conditional_dim (int): size of the conditional latent representation.
                    n_layer (int): number of spectral layers to be used
                    num_model_layer(int): number of model layer
                    drop_out (float): dropout rate in [0, 1).
                    activation (str): string indicationg the activation function.
                    norm (Literal['layer', 'batch', 'none']): normalization layer to be used.
                    dtype (torch.dtype, optional): type of data. Defaults to torch.float64.
                    dft_scale (float, optional): Amount to scale dft signal. Defaults to 1.
                    dft_shift (float, optional): Amount to shift dft signal. Defaults to 0.
                seq_len : int
                epochs: int
                batch_size : int
                lag : int
                optim_config: config for adam optimizer (e.g. lr : float)
                lr_config : config for exponential lr_scheduler (e.g. gamma : float)
            cache (str | Path): cach path

        Returns:
            Dict[str, Any]: Metadata with keys:
                model_dict: mapping from symbols to models
                model_set: set of all models
                model_name:
                model_desc: description of the model
        """
        # set python, numpy, torch and cuda seeds
        set_seed(train_config["train_seed"])

        _MODEL_NAME = "cond_flow.pt"

        # check out cpu the main process is running on
        device = torch.device("cpu")
        cpu_name = cpuinfo.get_cpu_info()["brand_raw"]
        print("Conditional Flow Generator is using:")
        print(f"CPU : {cpu_name}")

        # initialize accelerator tool
        accelerator = Accelerator()
        device = accelerator.device

        cuda_name = None
        if train_config["use_cuda"] and torch.cuda.is_available():
            cuda_name = torch.cuda.get_device_name(device)
            print(f"GPU: {cuda_name}")

        # given dataframe determines the chosen symbol
        symbols = list(price_data.columns)

        if not len(symbols) % 2 == 0:
            raise ValueError(
                "The number of symbols to train on must be even due to the network architecture."
            )

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
        gen = conditional_flow_generator.ConditionalFlowGenerator()
        model_desc = gen.fit(
            price_data=price_data, config=train_config, accelerator=accelerator
        )
        model_desc["symbols"] = symbols
        metadata["model_dict"] = {sym: _MODEL_NAME for sym in symbols}
        metadata["model_set"] = [_MODEL_NAME]
        metadata["fit_scores"] = model_desc["fit_score"]

        torch.save(model_desc, cache / _MODEL_NAME)

        return metadata

    def sample(
        self,
        sample_config: Dict[str, Any],
        metadata: Dict[str, any],
        cache: str | Path,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            Tuple[pd.DataFrame, pd.Dataframe] : price_data, return_data
        """

        # set python, numpy, torch and cuda seeds
        set_seed(sample_config["sample_seed"])

        cache = Path(cache)

        n_burn = sample_config["n_burn"]
        n_sample = sample_config["n_sample"]

        # read metadata
        model_dict = metadata["model_dict"]

        symbols = list(model_dict.keys())

        generator = conditional_flow_generator.ConditionalFlowGenerator()

        # take time index starting from today
        time_idx = pd.date_range(str(date.today()), periods=n_sample + 1, freq="D")
        generated_prices = pd.DataFrame(
            np.zeros((n_sample + 1, len(symbols))), columns=symbols, index=time_idx
        )
        generated_returns = pd.DataFrame(
            np.zeros((n_sample, len(symbols))), columns=symbols, index=time_idx[1:]
        )

        if len(metadata["model_set"]) > 1:
            warn("There is more than one model in the model set apparently.")
        model_config = torch.load(cache / metadata["model_set"][0])
        price_simulation, return_simulation = generator.sample(
            n_sample, burn=n_burn, config=model_config
        )
        generated_prices.loc[:, :] = price_simulation
        generated_returns.loc[:, :] = return_simulation

        return generated_prices, generated_returns
