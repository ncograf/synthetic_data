from datetime import date
from pathlib import Path
from typing import Any, Dict, Tuple

import base_index_generator
import garch_univar_generator
import numpy as np
import pandas as pd
import torch
from accelerate.utils import set_seed


class GarchIndexGenerator(base_index_generator.BaseIndexGenerator):
    def __init__(self):
        self.model_desc = "Univariate GARCH model"
        self.model_name = "univar_garch_model"

    def fit(
        self,
        price_data: pd.DataFrame,
        train_config: Dict[str, any],
        cache: str | Path,
    ) -> Dict[str, Any]:
        """For each column in price data fit a GARCH model and store the models in the cache

        Args:
            price_data (pd.DataFrame): Price data, where column indices are regarded as the chosen stocks
            train_conifg (Dict[str, any]): Fitting configuration for the GARCH models must contain:
                'train_seed' : integer seed for training
                'garch_config' :
                    'q' : integer from standard GARCH
                    'p' : integer from standard GARCH
                    'dist' : distribution ['normal', 'studentt']
            cache (str | Path): cache path
            **kwargs : ignored

        Returns:
            Dict[str, any]: metadata
                'model_dict': mapping from symbols -> file-name in `cache`
                'model_set': list of model file names
        """

        set_seed(train_config["train_seed"])

        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)

        # get model configuration for training
        model_config = train_config["garch_config"]  # p, q, dist

        # accumulate metadata for fit artifact
        metadata = {
            "model_dict": {},
            "model_set": set(),
            "model_desc": self.model_desc,
            "model_name": self.model_name,
        }

        symbols = price_data.columns
        generator = garch_univar_generator.GarchUnivarGenerator()

        log_likelyhood = []
        # sample all indices
        for sym in symbols:
            temp_dict = generator.fit_model(
                price_data=price_data.loc[:, sym], config=model_config
            )
            temp_dict["symbol"] = sym
            metadata["model_dict"][sym] = f"{sym}.pt"
            metadata["model_set"].add(f"{sym}.pt")
            log_likelyhood.append(temp_dict["fit_score"])

            torch.save(temp_dict, cache / f"{sym}.pt")

        metadata["fit_scores"] = log_likelyhood

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

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame] : Sampled Prices, Sampled Returns
        """

        set_seed(sample_config["sample_seed"])

        cache = Path(cache)

        n_burn = sample_config["n_burn"]
        n_sample = sample_config["n_sample"]

        # read metadata
        model_dict = metadata["model_dict"]

        symbols = list(model_dict.keys())

        generator = garch_univar_generator.GarchUnivarGenerator()

        # take time index starting from today
        time_idx = pd.date_range(str(date.today()), periods=n_sample + 1, freq="D")
        generated_prices = pd.DataFrame(
            np.zeros((n_sample + 1, len(symbols))), columns=symbols, index=time_idx
        )
        generated_returns = pd.DataFrame(
            np.zeros((n_sample, len(symbols))), columns=symbols, index=time_idx[1:]
        )

        for sym in symbols:
            # model_config with 'garch' : fitted model description (mu, omega, alpha ..)
            #                   'init_price' : initial price
            model_config = torch.load(cache / model_dict[sym])
            price_simulation, return_simulation = generator.sample(
                n_sample, burn=n_burn, config=model_config
            )
            generated_prices.loc[:, sym] = price_simulation
            generated_returns.loc[:, sym] = return_simulation

        return generated_prices, generated_returns
