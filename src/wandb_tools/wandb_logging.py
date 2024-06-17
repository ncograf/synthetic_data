from pathlib import Path

import pandas as pd
import visualize_stylized_facts as vst
import visualize_temp_series as vts

import wandb


def log_stylized_facts(
    local_path: str | Path,
    wandb_path: str,
    figure_title: str,
    price_data: pd.DataFrame,
    return_data: pd.DataFrame,
):
    """Log stylized facts and catch errors (errors will be printed but not thrown)

    Args:
        local_path (str | Path): local path to store image of stylized facts
        wandb_path (str): wandb path to store stylized facts
        figure_title (str): title of figure with stylized facts
        price_data (pd.DataFrame): price data
        return_data (pd.DataFrame): return data
    """

    # compute stylized facts
    try:
        plot = vst.visualize_stylized_facts(price_data, return_data)
        plot.suptitle(figure_title)
        plot.savefig(local_path)
        if wandb.run is not None:
            image = wandb.Image(plot, caption=figure_title)
            wandb.log({wandb_path: image})
    except Exception as e:
        raise e


def log_temp_series(
    local_path: str | Path, wandb_path: str, figure_title: str, temp_data: pd.DataFrame
):
    """Log temporal series and catch errors (errors will be printed but not thrown)

    Args:
        local_path (str | Path): local path to store image of stylized facts
        wandb_path (str): wandb path to store stylized facts
        figure_title (str): title of figure with stylized facts
        temp_data (pd.DataFrame): temporal data
    """

    # compute stylized facts
    try:
        plot = vts.visualize_temp_data(temp_data)
        plot.suptitle(figure_title)
        plot.savefig(local_path)
        if wandb.run is not None:
            image = wandb.Image(plot, caption=figure_title)
            wandb.log({wandb_path: image})
    except Exception as e:
        raise e
