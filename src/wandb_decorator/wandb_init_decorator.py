import functools
import inspect
import os
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
import stylized_facts_visualizer
import tail_dist_visualizer
import temporal_series_visualizer

import wandb


def setup_wandb_api(wandb_off: bool):
    if wandb_off:
        os.environ["WANDB_MODE"] = "disabled"

    elif not os.getenv("WANDB_MODE") == "disabled":
        if os.getenv("WANDB_API_KEY") is None:
            raise EnvironmentError("WANDB_API_KEY is not set!")

        if os.getenv("WANDB_ENTITY") is None:
            raise EnvironmentError("WANDB_ENTITY is not set!")

        if os.getenv("WANDB_PROJECT") is None:
            raise EnvironmentError("WANDB_PROJECT is not set!")


def wandb_run(func):
    """Runs function in a wandb run
    Can be turned off with the function argument `wandb_off`
    or by setting the environment variable WANDB_MODE='disabled'
    """

    @functools.wraps(func)  # type hints? / inception
    def wrapper(*args, **kwargs):
        if "wandb_off" in kwargs.keys():
            setup_wandb_api(wandb_off=kwargs["wandb_off"])
        else:
            # run wandb by default
            setup_wandb_api(wandb_off=False)

        # wrap function in a wandb run
        with wandb.init():
            result = func(*args, **kwargs)

        return result

    return wrapper


def wandb_fit(func):
    """Log a training run to wandb. To work, the decorated
    function needs to have a list of arguments as given by
    the BaseIndexGenerator fit function.

    Args:
        func (callable): Function with specific arguments
    """

    _DATA_PARAM = "price_data"
    _DATA_DESC = "Time series for stocks which has to be sliced."
    _CACHE_PARAM = "cache"
    _CONFIG_PARAM = "training_config"
    _SEED_PARAM = "seed"

    # get function signature
    sig = inspect.signature(func)

    # check for correct function signature
    if not {_DATA_PARAM, _CACHE_PARAM, _CONFIG_PARAM, _SEED_PARAM}.issubset(
        sig.parameters
    ):
        raise TypeError(
            f"Function {func.__name__} cannot be wrapped (wrong arguments)!"
        )

    # Real wrapper function
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # log data if given as an argument
        ba = sig.bind(*args, **kwargs)

        data = ba.arguments.get(_DATA_PARAM)
        cache_dir = Path(ba.arguments.get(_CACHE_PARAM))
        config = ba.arguments.get(_CONFIG_PARAM)
        seed = ba.arguments.get(_SEED_PARAM)

        # update wandb config
        wandb.config[_CONFIG_PARAM] = config
        wandb.config[_SEED_PARAM] = seed

        # store dataset
        data_artifact = wandb.Artifact(
            name=_DATA_PARAM,
            type="dataset",
            description=_DATA_DESC,
        )
        data_path = cache_dir / "data.parquet"
        data.to_parquet(data_path, compression="gzip")

        # log locally stored dataset
        data_artifact.add_file(local_path=data_path, name="datasets/train_data.parquet")
        wandb.log_artifact(data_artifact)

        metadata = func(*args, **kwargs)

        if not {"model_dict", "model_set", "model_desc", "model_name"}.issubset(
            metadata
        ):
            raise TypeError(
                (
                    f"Function {func.__name__}"
                    "does not return a valid output "
                    "(missing arguments in metadata)!"
                )
            )

        model_name = metadata["model_name"]
        model_artifact = wandb.Artifact(
            name=model_name, type="model", metadata=metadata
        )

        wandb.tags = wandb.tags + [model_name]

        # store all models in the model set
        for name in metadata["model_set"]:
            path = cache_dir / name
            if path.exists():
                model_artifact.add_file(path, name=f"{model_name}/{name}")
            else:
                warn(f"Model {name} was not found in {str(path)}.")

        return metadata

    return wrapper


def wandb_eval(func):
    """Evaluation function to wrap the sample function.
    This will store the result to wandb

    Args:
        func: Function with parameter requirements
    """

    _N_SAMPLE_PARAM = "n_samples"
    _METADATA_PARAM = "metadata"
    _CACHE_PARAM = "cache"
    _SEED_PARAM = "seed"
    _REAL_DATA_PARAM = "real_data"

    # get function signature
    sig = inspect.signature(func)

    # check for correct function signature
    if not {_N_SAMPLE_PARAM, _CACHE_PARAM, _METADATA_PARAM, _SEED_PARAM}.issubset(
        sig.parameters
    ):
        raise TypeError(
            f"Function {func.__name__} cannot be wrapped (wrong arguments)!"
        )

    if _REAL_DATA_PARAM in set(sig.parameters):
        raise TypeError(
            f"Function {func.__name__} already takes the argument {_REAL_DATA_PARAM}, cannot be wrapped!"
        )

    # Real wrapper function
    @functools.wraps(func)
    def wrapper(*args, real_data: pd.DataFrame, **kwargs):
        # log data if given as an argument
        ba = sig.bind(*args, **kwargs)

        seed = ba.arguments.get(_SEED_PARAM)
        n_samples = ba.arguments.get(_N_SAMPLE_PARAM)

        # update wandb config
        wandb.config["sample_seed"] = seed
        wandb.config["n_samples"] = n_samples

        generated_prices, generated_returns = func(*args, **kwargs)

        plot_price = temporal_series_visualizer.visualize_time_series(
            generated_prices, [{"linestyle": "-", "linewidth": 1}], "price-simulation"
        )
        wandb.log({"price simulation plot": plot_price.figure})

        plot_return = temporal_series_visualizer.visualize_time_series(
            generated_returns, [{"linestyle": "-", "linewidth": 1}], "return-simulation"
        )
        wandb.log({"return simulation plot": plot_return.figure})

        # compute stylized facts
        plot = stylized_facts_visualizer.visualize_all(
            stock_data=generated_prices, name="Stylized Facts simulation."
        )
        image = wandb.Image(plot.figure, caption="Stylized Facts")
        wandb.log({"stylized_facts": image})

        gen_log_returns = np.log(generated_returns)
        log_returns = np.log(real_data[1:] / real_data[:-1])
        plot = tail_dist_visualizer.visualize_tail(
            real_series=log_returns,
            time_series=gen_log_returns,
            plot_name="Tail Statistics",
            quantile=0.01,
        )
        image = wandb.Image(plot.figure, caption="Tail Statistics")
        wandb.log({"tail_stat": image})

        return generated_prices, generated_returns

    return wrapper


# TODO TEST, FINISH, AND USE
def get_train_run(train_run: str, verbose: bool):
    api = wandb.Api()

    setup_wandb_api(wandb_off=False)

    runs = api.runs(
        path=f'{os.getenv("WANDB_ENTITY")}/{os.getenv("WANDB_PROJECT")}',
        filters={"name": train_run},
    )
    if not len(runs) == 1:
        runs = api.runs(
            path=f'{os.getenv("WANDB_ENTITY")}/{os.getenv("WANDB_PROJECT")}',
            filters={"displayName": train_run},
        )
    elif verbose:
        print(f"Run fund for RUN_ID: {train_run}.")

    if len(runs) == 0:
        raise RuntimeError(f"No runs found on wandb for run_id / run_name {train_run}!")
    elif len(runs) > 1:
        run_list = ["/".join(r.path) for r in runs]
        run_list_str = "\n".join(run_list)
        raise RuntimeError(
            f"Multiple runs found for run name {train_run}:\n{run_list_str}"
        )

    trained_run = runs[0]
    model_artifacts = [a for a in trained_run.logged_artifacts() if a.type == "model"]
    dataset_artifacts = [
        a for a in trained_run.logged_artifacts() if a.type == "dataset"
    ]

    if len(model_artifacts) > 1:
        print(
            f"The train run {trained_run.name} contains more than one model, only the first one is chosen."
        )
    if len(dataset_artifacts) > 1:
        print(
            f"The train run {trained_run.name} contains more than one dataset, the first one is chosen."
        )
    if len(dataset_artifacts) == 0:
        raise RuntimeError(
            f"The train run {trained_run.name} does not contain a dataset."
        )

    # TODO download artifact and return metadata
    try:
        # get model
        model_artifact = model_artifacts[0]
        # model_dir = model_artifact.download()
        # model_artifact_data = torch.load(Path(model_dir) / "conditional_flow.pt")

        dataset_dir = dataset_artifacts[0].download()
        dataset_path = Path(dataset_dir) / "datasets/price_data.parquet"
        price_data = pd.read_parquet(dataset_path)

    except wandb.errors.CommError as err:
        print(f"{err.message}")
        return

    return model_artifact.metadata, price_data
