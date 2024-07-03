from pathlib import Path
from typing import Any, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy.typing as npt
import pandas as pd
import torch
import visualize_stylized_facts as vst
import visualize_temp_series as vts

import wandb


def log_stylized_facts(
    local_path: str | Path,
    wandb_path: str,
    figure_title: str,
    log_returns: npt.ArrayLike,
):
    """Log stylized facts and catch errors (errors will be printed but not thrown)

    Args:
        local_path (str | Path): local path to store image of stylized facts
        wandb_path (str): wandb path to store stylized facts
        figure_title (str): title of figure with stylized facts
        log_returns (npt.): return data
    """

    matplotlib.use("Agg")

    # compute stylized facts
    try:
        plot = vst.visualize_stylized_facts(log_returns=log_returns)
        plot.suptitle(figure_title)
        plot.savefig(local_path)
        if wandb.run is not None:
            image = wandb.Image(plot, caption=figure_title)
            wandb.log({wandb_path: image})
        plot.clear()
        plt.close(plot)
    except Exception as e:
        print(f"Expeption occured on logging stylized facts: {str(e)}.")
    finally:
        plt.clf()
        plt.close("all")  # works as long as the code is seqential


def log_temp_series(
    local_path: str | Path, wandb_path: str, figure_title: str, temp_data: npt.ArrayLike
):
    """Log temporal series and catch errors (errors will be printed but not thrown)

    Args:
        local_path (str | Path): local path to store image of stylized facts
        wandb_path (str): wandb path to store stylized facts
        figure_title (str): title of figure with stylized facts
        temp_data (npt.ArrayLike): temporal data
    """
    matplotlib.use("Agg")

    try:
        plot = vts.visualize_temp_data(temp_data)
        plot.suptitle(figure_title)
        plot.savefig(local_path)
        if wandb.run is not None:
            image = wandb.Image(plot, caption=figure_title)
            wandb.log({wandb_path: image})
        plot.clear()
        plt.close(plot)
    except Exception as e:
        print(f"Expeption occured on logging temporal series: {str(e)}.")
    finally:
        plt.clf()
        plt.close("all")  # works as long as the code is seqential


def log_train_data(
    local_path: str | Path,
    wandb_name: str,
    data: pd.DataFrame,
):
    """Log data from pandas DataFrame and catch errors (errors will be printed but not thrown)

    Args:
        local_path (str | Path): local path to store image of stylized facts
        wandb_name (str): path / name on wandb
        data (pd.DataFrame): data to log as parquet file
    """

    try:
        # store data
        data.to_parquet(local_path, compression="gzip")
        if wandb.run is not None:
            data_artifact = wandb.Artifact(
                name=wandb_name,
                type="dataset",
                description="Raw data used for training (note that this data might need to be processed).",
            )
            data_artifact.add_file(local_path=local_path, name=f"{wandb_name}.parquet")
            wandb.log_artifact(data_artifact)
    except Exception as e:
        print(f"Expeption occured on logging data: {str(e)}.")


def log_model(
    model_dict: Dict[str, Any],
    wandb_name: str,
    local_path: str | Path,
    desc: str,
):
    """Log model to wandb and locally and catch error (erros will be printed but not thrown)

    Args:
        model_dict (Dict[str, Any]): Model container including:
            - state_dict : model state
            - init_params : model initialization params
            - network : model to string
            - fit_score : any score representing 'goodness of the fit'
        wandb_name (str): name to use on wandb
        local_path (str | Path): local path on computer
        desc (str): model
    """

    try:
        # store metadata for simpler / automatic restoring
        artifact_wandb_path = f"{Path(wandb_name).stem}.pt"
        meta_data = {"model_path": artifact_wandb_path, "description": desc}
        model_artifact = wandb.Artifact(
            name=wandb_name, type="model", metadata=meta_data
        )

        # save model
        torch.save(model_dict, local_path)
        model_artifact.add_file(local_path, name=artifact_wandb_path)

        wandb.log_artifact(model_artifact)
    except Exception as e:
        print(f"Expeption occured on logging the model: {str(e)}.")


def load_local_model(model_path: str | Path) -> Dict[str, Any]:
    """Loads a model from a saved place

    Args:
        model_path (str | Path): Path to the model file with the following dictionary
                - state_dict : model state
                - init_params : model initialization params
                - network : model to string
    Returns:
        Dict[str, Any]: the dictonary stored in the model
    """

    try:
        model_dict = torch.load(model_path)
        return model_dict

    except Exception as e:
        print(f"Expeption occured on loading the model locally: {str(e)}.")
