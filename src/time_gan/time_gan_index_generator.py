import time
from datetime import date
from pathlib import Path
from typing import Dict, List

import cpuinfo
import numpy as np
import pandas as pd
import stylized_facts_visualizer
import tail_dist_visualizer
import temporal_series_visualizer
import time_gan_generator
import torch
from accelerate import Accelerator
from time_gan import TimeGan

import wandb
import wandb.errors


class TimeGanIndexGenerator:
    def __init__(self, cache: str | Path = "data/cache"):
        self.cache = Path(cache) / "tfig"
        self.cache.mkdir(parents=True, exist_ok=True)

    def fit_local(
        self,
        generator: time_gan_generator.TimeGanGenerator,
        data: pd.Series,
        kwargs: Dict[str, any],
    ):
        generator.fit_model(price_data=data, **kwargs)

    def fit_index(
        self,
        price_index_data: pd.DataFrame,
        train_config: Dict[str, any],
        tags: List[str] = [],
        notes: str = "",
        dtype: str = "float32",
        use_cuda: bool = True,
    ):
        """Fit data for the given input data

        Args:
            price_index_data (pd.DataFrame): Data to fit model on, all columns will be fitted.
            train_conifg (Dict[str, any]): Config used for training
            tags (List[str]): Tags to be associated with the run for wandb
            notes (str): Notes about the run
            dtype (str): dtype to use when training
            use_cuda (bool): Whether or not to use cuda if available
        """

        # check out cpu the main process is running on
        self.device = torch.device("cpu")
        self.cpu_name = cpuinfo.get_cpu_info()["brand_raw"]
        print("Fourier Flow Generator is using:")
        print(f"CPU : {self.cpu_name}")

        # initialize accelerator tool
        accelerator = Accelerator()
        self.device = accelerator.device

        self.cuda_name = None
        if use_cuda and torch.cuda.is_available():
            self.cuda_name = torch.cuda.get_device_name(self.device)
            print(f"GPU: {self.cuda_name}")

        symbols = list(price_index_data.columns)

        general_config = {
            "architecture": "time gan",
            "cpu": self.cpu_name,
            "gpu": self.cuda_name,
            "device": str(self.device),
            "dtype": dtype,
        }
        general_config = general_config | train_config

        run = wandb.init(
            project="synthetic_data",
            entity="ncograf",
            group="time_gan",
            job_type="train",
            tags=["base_time_gan"] + symbols + tags,
            notes=notes,
            config=general_config,
        )

        run.define_metric("*", step_metric="epoch")

        # store dataset
        data_artifact = wandb.Artifact(
            name="price_data",
            type="dataset",
            description="Price data from which to compute training seqences",
        )
        data_name = self.cache / "price_index_data.parquet"
        price_index_data.to_parquet(data_name, compression="gzip")
        data_artifact.add_file(
            local_path=data_name, name="datasets/price_index_data.parquet"
        )
        run.log_artifact(data_artifact)

        train_config["run"] = run
        train_config["accelerator"] = accelerator

        for sym in symbols:
            data = price_index_data.loc[:, sym]
            gen = time_gan_generator.TimeGanGenerator(
                symbol=sym, name="TimeGan", cache=self.cache, dtype=dtype
            )
            gen.fit_model(price_data=data, **train_config)

    def sample_wandb_index(
        self,
        train_run: str,
        notes: str = "",
        sample_len: int = 2000,
        verbose: bool = False,
        run_by_id: bool = False,
    ) -> pd.DataFrame:
        """Generates synthetic data for the whole index

        Args:
            train_run (str): Trained model to be chosen from train run on wandb
            notes (str, optional): Notes to be associated with the sampling run. Defaults to "".
            sample_len (int, optional): Seqence length to be sampled. Defaults to 2000.
            verbose (bool, optional): Print output. Defaults to False.
            run_by_id (bool, optional): Set true if train_run is an id instead of the run name. Defaults to False.

        Raises:
            ValueError: data must be a DataFrame

        Returns:
            pd.DataFrame : DataFrame containing all the columns
        """

        api = wandb.Api()
        if run_by_id:
            runs = api.runs(path="ncograf/synthetic_data", filters={"name": train_run})
        else:
            runs = api.runs(
                path="ncograf/synthetic_data", filters={"displayName": train_run}
            )

        if len(runs) == 0:
            raise RuntimeError(
                f"No runs found on wandb for {'run_id' if run_by_id else 'run_name'} {train_run}!"
            )
        elif len(runs) > 1:
            run_list = ["/".join(r.path) for r in runs]
            run_list_str = "\n".join(run_list)
            raise RuntimeError(
                f"Multiple runs found for run name {train_run}:\n{run_list_str}"
            )

        trained_run = runs[0]
        model_artifacts = [
            a for a in trained_run.logged_artifacts() if a.type == "model"
        ]
        dataset_artifacts = [
            a for a in trained_run.logged_artifacts() if a.type == "dataset"
        ]

        if len(dataset_artifacts) > 1:
            print(
                f"The train run {trained_run.name} contains more than one dataset, the first one is chosen."
            )
        if len(dataset_artifacts) == 0:
            raise RuntimeError(
                f"The train run {trained_run.name} does not contain a dataset."
            )

        dataset_dir = dataset_artifacts[0].download()
        dataset_path = Path(dataset_dir) / "datasets/price_index_data.parquet"
        price_data = pd.read_parquet(dataset_path)

        symbols = [a.metadata["symbol"] for a in model_artifacts]

        time_idx = pd.date_range(str(date.today()), periods=sample_len, freq="D")
        generated_returns = pd.DataFrame(
            data=np.zeros((sample_len, len(symbols))),
            columns=symbols,
            index=time_idx,
        )
        generated_prices = pd.DataFrame(
            data=np.zeros((sample_len, len(symbols))),
            columns=symbols,
            index=time_idx,
        )
        self._problematic_cols = []

        run = wandb.init(
            project="synthetic_data",
            entity="ncograf",
            group="base_time_gan",
            job_type="generate_data",
            tags=["base_time_gan"],
            notes=notes,
            config={
                "len": sample_len,
                "burn": 0,
                "train_run_path": "/".join(trained_run.path),
                "train_run_name": trained_run.name,
                "symbols": symbols,
            },
        )

        t = time.time()
        for artifact in model_artifacts:
            try:
                # get model
                run.use_artifact(artifact)
                model_dir = artifact.download()
                symbol = artifact.metadata["symbol"]
                model_artifact_data = torch.load(Path(model_dir) / f"{symbol}.pth")

            except wandb.errors.CommError as err:
                print(f"{err.message}")
                run.finish(1)  # exit code 1 for error
                return

            # print this or use it to see what was downloaded
            # description = model_artifact.metadata

            model = TimeGan(**model_artifact_data["init_params"])
            model.load_state_dict(model_artifact_data["state_dict"])
            accelerator = Accelerator()
            model = accelerator.prepare(model)

            scale = model_artifact_data["scale"]
            shift = model_artifact_data["shift"]
            init_price = model_artifact_data["init_price"]
            generator = time_gan_generator.TimeGanGenerator(symbol=symbol)
            price_simulation, return_simulation = generator.generate_data(
                model=model,
                scale=scale,
                shift=shift,
                init_price=init_price,
                len_=sample_len,
                burn=100,
            )

            if np.isinf(price_simulation).any():
                self._problematic_cols.append(symbol)

            generated_prices.loc[:, symbol], generated_returns.loc[:, symbol] = (
                price_simulation,
                return_simulation,
            )

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
        log_returns = np.log(price_data[1:] / price_data[:-1])
        plot = tail_dist_visualizer.visualize_tail(
            real_series=log_returns,
            time_series=gen_log_returns,
            plot_name="Tail Statistics",
            quantile=0.01,
        )
        image = wandb.Image(plot.figure, caption="Tail Statistics")
        wandb.log({"tail_stat": image})

        if verbose:
            print(f"Sampling took {t - time.time()} seconds.")
            print(f"Stocks containing infinity : {self._problematic_cols}.")

        return generated_prices, generated_returns
