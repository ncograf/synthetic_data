from pathlib import Path
from typing import Any, Dict

import base_index_generator
import pandas as pd
import wandb_decorator


def wandb_fit(
    index_generator: base_index_generator.BaseIndexGenerator,
    price_data: pd.DataFrame,
    train_config: Dict[str, any],
    cache: str | Path,
    **kwargs,
):
    """Fit the given index generator with logging on wandb

    Args:
        index_generator (base_index_generator.BaseIndexGenerator): generator to use
        price_data (pd.DataFrame): price data to train on
        train_config (Dict[str, any]): training dictonary to be fed into the generators fit function
        cache (str | Path): path to local cache

    Returns:
        Dict[str, Any]: metadata with description of the fitted model
    """

    @wandb_decorator.wandb_fit
    def _fit(
        price_data: pd.DataFrame,
        train_config: Dict[str, any],
        cache: str | Path,
        **kwargs,
    ):
        return index_generator.fit(price_data, train_config, cache, **kwargs)

    metadata = _fit(
        price_data=price_data, train_config=train_config, cache=cache, **kwargs
    )
    return metadata


def wandb_eval(
    index_generator: base_index_generator.BaseIndexGenerator,
    sample_config: Dict[str, Any],
    metadata: Dict[str, Any],
    cache: str | Path,
    **kwargs,
):
    """Eval method with wandb logging

    Args:
        index_generator : base_index_generator.BaseIndexGenerator
        sample_config (Dict[str, Any]): configuration to be fed into the generators sampler
        metadata (Dict[str, Any]): model description data
        cache (str | Path): cache where the model is stored

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Sampled Prices and indices
    """

    @wandb_decorator.wandb_eval
    def _eval(
        sample_config: Dict[str, Any],
        metadata: Dict[str, Any],
        cache: str | Path,
        **kwargs,
    ):
        return index_generator.sample(sample_config, metadata, cache, **kwargs)

    simulations = _eval(
        sample_config=sample_config, metadata=metadata, cache=cache, **kwargs
    )
    return simulations


def wandb_train(
    index_generator: base_index_generator.BaseIndexGenerator,
    price_data: pd.DataFrame,
    train_config: Dict[str, Any],
    cache: str | Path,
    **kwargs,
):
    """Train run of index generator with wandb logging and integrated evaluation

    This function can be used to start runs outside of a sweep environment

    Args:
        index_generator (base_index_generator.BaseIndexGenerator): generator to be fitted
        price_data (pd.DataFrame): price data to fit on
        train_config (Dict[str, Any]): training configuration according to the generator
            including `sample_config` key
            including `train_config` key
        cache (str | Path): local cache directory
    """

    @wandb_decorator.wandb_run
    def _train(price_data, train_config, cache, **kwargs):
        metadata = wandb_fit(
            index_generator, price_data, train_config["train_config"], cache, **kwargs
        )

        wandb_eval(
            index_generator,
            sample_config=train_config["sample_config"],
            metadata=metadata,
            cache=cache,
            real_data=price_data,  # note that the decorator needs this extra argument
        )

    _train(price_data, train_config, cache, **kwargs)
