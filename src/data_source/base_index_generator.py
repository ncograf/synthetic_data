from abc import abstractmethod
from pathlib import Path
from typing import Dict, Tuple

import numpy.typing as npt
import pandas as pd


class BaseIndexGenerator:
    @abstractmethod
    def fit(
        self,
        price_data: pd.DataFrame,
        training_config: Dict[str, any],
        cache: str | Path,
        seed: int,
        **kwargs,
    ):
        """Fit a model for the given training configuration and store the model in the cache

        Args:
            price_data (pd.DataFrame): Price data, where column indices are regarded as the chosen stocks
            training_conifg (Dict[str, any]): Fitting configuration for the model(s)
            cache (str | Path): cache path
            seed (int): fix randomness

        Returns:
            Dict[str, any]: metadata description of the models storage location
        """
        raise NotImplementedError(
            "This function must be implemented by the class which inherits"
        )

    @abstractmethod
    def sample(
        self,
        n_samples: int,
        metadata: Dict[str, any],
        cache: str | Path,
        seed: int,
        **kwargs,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generates synthetic data for the whole index

        Args:
            n_samples (int): Number of samples
            metadata (Dict[str, any]): Dictonary describing the filenames of the model(s)
            cache (str | Path): Path to the stored generators / models
            seed (int): Seed to make the sampling reproducible

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame] : Sampled Prices, Sampled Returns
        """
        raise NotImplementedError(
            "This function must be implemented by the class which inherits"
        )
