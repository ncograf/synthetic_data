import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import fourier_flow_generator
import numpy as np
import pandas as pd
import torch
import utils.type_converter as type_converter
import visualizer
from fourier_flow import FourierFlow

import wandb
import wandb.errors


class FourierFlowIndexGenerator:
    def __init__(self):
        pass

    def fit_index(
        self,
        price_index_data: pd.DataFrame,
        fit_params: Dict[str, any] = {},
        dtype: str = "float32",
    ):
        """Fit data for the given input data

        Args:
            price_index_data (pd.DataFrame): Data to fit model on, all columns will be fitted.
            fit_params (Dict[str, any], optional): Extra parameters used for the fit function. Defaults to {}.
            dtype (str): Float
        """

        columns = price_index_data.columns

        for col in columns:
            series = price_index_data.loc[:, col]
            mask = ~np.isnan(series)

            generator = fourier_flow_generator.FourierFlowGenerator(
                symbol=col, dtype=dtype
            )
            generator.fit_model(series.loc[mask], **fit_params)

    def sample_wandb_index(
        self,
        model_info: List[Tuple[str, str, str]],
        dtype: str = "float32",
        notes: str = "",
        sample_len: int = 2000,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Generates synthetic data for the whole index

        Args:
            data (pd.DataFrame): All data in the index
            model_info (List[Tuple[str, str, str]]): Model (artifact identifier, symbol, version) containing information to pull model from wandb.
            n_cpu (int): Number of cpu's to use (note the actual number might deviate based on machine limits).
            verbose (bool): Print output.

        Raises:
            ValueError: data must be a DataFrame

        Returns:
            pd.DataFrame : DataFrame containing all the columns
        """

        _, symbols, versions = tuple(zip(*model_info))

        time_idx = pd.date_range(str(date.today()), periods=sample_len, freq="D")
        generated_returns = pd.DataFrame(
            data=np.zeros((sample_len, len(model_info))),
            columns=symbols,
            index=time_idx,
        )
        generated_prices = pd.DataFrame(
            data=np.zeros((sample_len, len(model_info))),
            columns=symbols,
            index=time_idx,
        )
        self._problematic_cols = []

        run = wandb.init(
            project="synthetic_data",
            entity="ncograf",
            group="base_fourier_flow",
            job_type="generate_data",
            tags=["base_fourier_flows"],
            notes=notes,
            config={
                "len": sample_len,
                "burn": 0,
                "model_version": versions,
                "symbols": symbols,
            },
        )

        t = time.time()
        for identifier, symbol, version in model_info:
            try:
                model_artifact = run.use_artifact(
                    f"fourier_flow_model_{symbol}:{version}"
                )
                model_dir = model_artifact.download()
                model_artifact_data = torch.load(Path(model_dir) / "model_state.pth")
            except wandb.errors.CommError as err:
                print(f"{err.message}")
                run.finish(1)  # exit code 1 for error
                return

            # print this or use it to see what was downloaded
            # description = model_artifact.metadata

            model = FourierFlow(**model_artifact_data["init_params"])
            model.load_state_dict(model_artifact_data["state_dict"])

            model.dtype = type_converter.TypeConverter.str_to_torch(dtype)

            scale = model_artifact_data["scale"]
            shift = model_artifact_data["shift"]
            init_price = model_artifact_data["init_price"]
            generator = fourier_flow_generator.FourierFlowGenerator(
                symbol=symbol, dtype=dtype
            )
            price_simulation, return_simulation = generator.generate_data(
                model=model,
                scale=scale,
                shift=shift,
                init_price=init_price,
                len_=sample_len,
            )

            if np.isinf(price_simulation).any():
                self._problematic_cols.append(symbol)

            generated_prices.loc[:, symbol], generated_returns.loc[:, symbol] = (
                price_simulation,
                return_simulation,
            )

            price_table = np.stack(
                [price_simulation, np.arange(price_simulation.shape[0])], axis=1
            )
            price_table = wandb.Table(
                data=price_table, columns=["stock price", "timesteps"]
            )
            return_table = np.stack(
                [return_simulation, np.arange(return_simulation.shape[0])], axis=1
            )
            return_table = wandb.Table(
                data=return_table, columns=["stock price returns", "timesteps"]
            )
            run.log(
                {
                    f"price_simulation_{symbol}": wandb.plot.line(
                        price_table,
                        x="timesteps",
                        y="stock price",
                        title=f"{symbol} Price Simulation",
                    ),
                    f"return_simulation_{symbol}": wandb.plot.line(
                        return_table,
                        x="timesteps",
                        y="stock price returns",
                        title=f"{symbol} Price Return Simulation",
                    ),
                }
            )

        # compute stylized facts
        print(generated_prices)
        plot = visualizer.visualize_all(
            stock_data=generated_prices, name="Stylized Facts simulation."
        )
        image = wandb.Image(plot.figure, caption="Stylized Facts")
        wandb.log({"stylized_facts": image})

        if verbose:
            print(f"Sampling took {t - time.time()} seconds.")
            print(f"Stocks containing infinity : {self._problematic_cols}.")

        return generated_prices, generated_returns
