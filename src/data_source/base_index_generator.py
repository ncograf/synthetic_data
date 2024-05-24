from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy.typing as npt
import pandas as pd


class BaseIndexGenerator:
    @abstractmethod
    def fit(
        self,
        price_data: pd.DataFrame,
        training_config: Dict[str, any],
        cache: str | Path,
        **kwargs,
    ):
        """Fit a model for the given training configuration and store the model in the cache

        Args:
            price_data (pd.DataFrame): Price data, where column indices are regarded as the chosen stocks
            training_conifg (Dict[str, any]): Fitting configuration for the model(s)
            cache (str | Path): cache path

        Returns:
            Dict[str, any]: metadata description of the models storage location
        """
        raise NotImplementedError(
            "This function must be implemented by the class which inherits"
        )

    @abstractmethod
    def sample(
        self,
        sample_config: Dict[str, Any],
        metadata: Dict[str, Any],
        cache: str | Path,
        **kwargs,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generates synthetic data for the whole index

        Args:
            sample_config (Dict[str, Any]): Config with seed and number of samples
            metadata (Dict[str, Any]): Dictonary describing the filenames of the model(s)
            cache (str | Path): Path to the stored generators / models

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame] : Sampled Prices, Sampled Returns
        """
        raise NotImplementedError(
            "This function must be implemented by the class which inherits"
        )
