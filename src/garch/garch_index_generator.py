from datetime import date
from pathlib import Path
from typing import Dict, Tuple

import base_index_generator
import garch_univar_generator
import numpy as np
import pandas as pd
import torch
from accelerate.utils import set_seed


class GarchIndexGenerator(base_index_generator.BaseIndexGenerator):
    def fit(
        self,
        price_data: pd.DataFrame,
        training_conifg: Dict[str, any],
        cache: str | Path,
        seed: int,
    ) -> Dict[str, any]:
        """For each column in price data fit a GARCH model and store the models in the cache

        Args:
            price_data (pd.DataFrame): Price data, where column indices are regarded as the chosen stocks
            training_conifg (Dict[str, any]): Fitting configuration for the GARCH models must contain:
                'garch_config' :
                    'q' : integer from standard GARCH
                    'p' : integer from standard GARCH
                    'dist' : distribution ['normal', 'studentt']
            cache (str | Path): cache path
            seed (int): fix randomness

        Returns:
            Dict[str, any]: metadata
                'model_dict': mapping from symbols -> file-name in `cache`
                'model_set': list of model file names
        """

        set_seed(seed=seed)

        cache = Path(cache)

        # get model configuration for training
        model_config = training_conifg["garch_config"]  # p, q, dist

        # accumulate metadata for fit artifact
        metadata = {"model_dict": {}, "model_set": set()}

        symbols = price_data.columns
        generator = garch_univar_generator.GarchUnivarGenerator()

        # sample all indices
        for sym in symbols:
            temp_dict = generator.fit_model(
                price_data=price_data.loc[:, sym], config=model_config
            )
            temp_dict["symbol"] = sym
            metadata["model_dict"][sym] = f"{sym}.pt"
            metadata["model_set"].add(f"{sym}.pt")

            torch.save(temp_dict, cache / f"{sym}.pt")

        return metadata

    def sample(
        self, n_samples: int, metadata: Dict[str, any], cache: str | Path, seed: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generates synthetic data for the whole index

        Args:
            n_samples (int): Number of samples
            metadata (Dict[str, any]): Dictonary describing the filenames of the model(s)
            cache (str | Path): Path to the stored generators / models
            seed (int): Seed to make the sampling reproducible

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame] : Sampled Prices, Sampled Returns
        """

        set_seed(seed)
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)

        n_burn = 300

        # read metadata
        model_dict = metadata["model_dict"]

        symbols = list(model_dict.keys())

        generator = garch_univar_generator.GarchUnivarGenerator()

        # take time index starting from today
        time_idx = pd.date_range(str(date.today()), periods=n_samples + 1, freq="D")
        generated_prices = pd.DataFrame(
            np.zeros((n_samples + 1, len(symbols))), columns=symbols, index=time_idx
        )
        generated_returns = pd.DataFrame(
            np.zeros((n_samples, len(symbols))), columns=symbols, index=time_idx[1:]
        )

        for sym in symbols:
            # model_config with 'garch' : fitted model description (mu, omega, alpha ..)
            #                   'init_price' : initial price
            model_config = torch.load(cache / model_dict[sym])
            price_simulation, return_simulation = generator.sample(
                n_samples, burn=n_burn, config=model_config
            )
            generated_prices.loc[:, sym] = price_simulation
            generated_returns.loc[:, sym] = return_simulation

        return generated_prices, generated_returns
