from pathlib import Path
from typing import List, Tuple

import base_generator
import cpuinfo
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from fourier_flow import FourierFlow
from torch.utils.data import DataLoader, TensorDataset
from type_converter import TypeConverter

import wandb
from wandb.sdk import wandb_run as wr


class FourierFlowGenerator(base_generator.BaseGenerator):
    def __init__(
        self,
        symbol: str = "NoName",
        name: str = "FourierFlow",
        dtype: str = "float64",
        cache: str | Path = "data/cache",
        use_cuda: bool = True,
    ):
        """Initialize Fourier Flow Generator

        Args:
            symbol (str, optional) : Name of the symbol to be created. Defaults to 'NoName'.
            name (str, optional): Generator name. Defaults to "FourierFlow".
            dtype (str, optional): dtype of model. Defaults to 'float64'.
            cache (str | Path, optional): cache location to store model and artifacts. Defaults to 'data/cache'.
            use_cuda (bool, optional): use cuda if available. Defaults to True.
        """
        base_generator.BaseGenerator.__init__(self, name)
        self.data_min = 0
        self.data_amplitude = 1

        if not isinstance(dtype, str):
            self.dtype_str = TypeConverter.type_to_str(dtype)
            self.torch_dtype = TypeConverter.str_to_torch(self.dtype_str)
            self.numpy_dtype = TypeConverter.str_to_numpy(self.dtype_str)
        else:
            self.dtype_str = TypeConverter.extract_dtype(dtype)
            self.torch_dtype = TypeConverter.str_to_torch(self.dtype_str)
            self.numpy_dtype = TypeConverter.str_to_numpy(self.dtype_str)

        self.cache = Path(cache) / "ff_train"
        self.cache.mkdir(parents=True, exist_ok=True)
        self._model: torch.Module | None = None

        self.device = torch.device("cpu")
        self.cpu_name = cpuinfo.get_cpu_info()["brand_raw"]
        print("Fourier Flow Generator is using:")
        print(f"CPU : {self.cpu_name}")

        self.cuda_mem = None
        self.cuda_name = None
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.cuda_name = torch.cuda.get_device_name()
            self.cuda_mem = torch.cuda.memory_allocated()
            print(f"GPU: {self.cuda_name}")
            print(f"Allocated CPU Memory: {self.cuda_mem}")

        self._symbol = symbol

    @property
    def symbol(self) -> str:
        return self._symbol

    @symbol.setter
    def symbol(self, value: str):
        self._symbol = value

    def fit_model(
        self,
        price_data: pd.DataFrame,
        learning_rate: float = 0.001,
        gamma: float = 0.995,
        epochs: int = 1000,
        batch_size: int = 128,
        hidden_dim: int = 200,
        num_layer: int = 10,
        lag: int = 1,
        seq_len: int = 101,
        notes: str = "",
        tags: List[str] = [],
    ):
        """Fit a Fourier Flow Neural Network. I.e. Train the network on the given data.

        Args:
            price_data (pd.DataFrame): Stock marked price datsa
            learning_rate (float, optional): Initial learing rate. Defaults to 0.001.
            gamma (float, optional): Exponential decay base of learning rate. Defaults to 0.995.
            epochs (int, optional): Number of training epochs. Defaults to 1000.
            batch_size (int, optional): Size of the batches to be processed. Defaults to 128.
            hidden_dim (int, optional): Hidden dimensions of the model recommended around 2 * seq_len. Defaults to 200.
            num_layer (int, optional): Number of layers in the Network. Defaults to 10.
            lag (int, optional): lag between the indivisual seqences when splitting the price. Defaults to 1.
            seq_len (int, optional): seqence lenght. Defaults to 101.
            notes (str, optional): notes describing the task. Defaults to "".
            tags (List[str], optional): List of tags. Defaults to [].

        Raises:
            ValueError: If the input is no a single price seqence
        """

        if price_data.ndim != 1:
            raise ValueError("Input price data must be one dimensional")

        general_config = {
            "architecture": "fourier_flow",
            "cpu": self.cpu_name,
            "gpu": self.cuda_name,
            "gpu_mem": self.cuda_mem,
            "seq_len": seq_len,
            "lag": lag,
        }

        config = {
            "hidden_dim": hidden_dim,
            "num_layer": num_layer,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "device": self.device,
        }

        # store configuration
        general_config = general_config | config

        run = wandb.init(
            project="synthetic_data",
            entity="ncograf",
            group="fourier_flow",
            job_type="train",
            tags=["base_fourier_flows", self.symbol] + tags,
            notes=notes,
            config=general_config,
        )

        # store dataset
        data_artifact = wandb.Artifact(
            name="train_data",
            type="dataset",
            description="Price data from which to compute training seqences",
        )
        data_name = self.cache / "price_data.csv"
        price_data.to_csv(data_name)
        data_artifact.add_file(local_path=data_name, name="datasets/price_data.csv")
        run.log_artifact(data_artifact)

        # transform dataset for training
        price_data = price_data.astype(self.numpy_dtype)
        data = torch.from_numpy(price_data.values)
        data_mask = ~torch.isnan(data)
        data = data[
            data_mask
        ]  # drop the nans TODO, this might give wrong results for missing nans
        self._zero_price = data[0]
        log_returns = torch.log(data[1:] / data[:-1])

        # split the data
        X = np.lib.stride_tricks.sliding_window_view(
            log_returns, seq_len, axis=0, writeable=False
        )
        X = torch.tensor(X[::lag, :])

        # train model
        self._model = self.train_fourier_flow(X=X, run=run, **config)

        # save model after training DO NOT CHANGE THE NAME as we need it when downloading the models
        model_path = self.cache / "model_state.pth"
        model_dict = {
            "state_dict": self._model.state_dict(),
            "init_params": self._model.get_model_info(),
            "network": str(self._model),
            "scale": self.data_amplitude,
            "shift": self.data_min,
            "init_price": self._zero_price,
        }
        torch.save(model_dict, model_path)
        model_artifact = wandb.Artifact(
            name=f"fourier_flow_model_{self.symbol}",
            type="model",
            metadata={
                "Info": "This information describes the features stored in the artifact",
                "state_dict": "Model weights to be loaded into a torch Module",
                "init_params": "Parameters to initialize network",
                "network": "Network class used in Training",
                "scale": "Scaling of the training data X_train = (X - shift) / scale",
                "shift": "Shift of the training data X_train = (X - shift) / scale",
                "init_price": "Initial Price for the stock for reconstruction",
            },
        )
        model_artifact.add_file(model_path)
        run.log_artifact(model_artifact)
        run.finish()

    def model(self) -> FourierFlow:
        return self._model

    def check_model(self):
        if self._model is None:
            raise RuntimeError("Model must bet set before.")

    def generate_data(
        self,
        len: int = 500,
        burn: int = 100,
        model_version: str | None = None,
        tags: List[str] = [],
        notes: str = "",
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generate data from the trained model

        If model_version = None, cosider to disable wandb with
        `os.environ['WANDB_MODE'] = 'disabled'`

        Args:
            len (int, optional): lengh of generated seqence. Defaults to 500.
            burn (int, optional): ignored here. Defaults to 100.
            model_version (str | None, optional): model, `latest` to use the latest model on wandb, None for local model is used. Defaults to None.
            tags (List[str], optional): list of tags for wandb. Defaults to [].
            notes (str, optional): Notes in markdown. Defaults to "".

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: price data and return data
        """

        run = wandb.init(
            project="synthetic_data",
            entity="ncograf",
            group="fourier_flow",
            job_type="generate_data",
            tags=["base_fourier_flows"] + tags,
            notes=notes,
            config={
                "len": len,
                "burn": burn,
                "model_version": model_version,
            },
        )

        if model_version is None:
            # use local model
            self.check_model()
            model = self._model
            scale = self.data_amplitude
            shift = self.data_min
            init_price = self._zero_price
        else:
            model_artifact = run.use_artifact(
                f"fourier_flow_model_{self.symbol}:{model_version}"
            )
            model_dir = model_artifact.download()
            model_artifact_data = torch.load(Path(model_dir) / "model_state.pth")

            # print this or use it to see what was downloaded
            # description = model_artifact.metadata

            model = FourierFlow(**model_artifact_data["init_params"])
            model.load_state_dict(model_artifact_data["state_dict"])

            model.dtype = self.torch_dtype  # use the current dtype

            scale = model_artifact_data["scale"]
            shift = model_artifact_data["shift"]
            init_price = model_artifact_data["init_price"]

        n = len // model.T + 1

        model.to(self.device)

        model_output = model.sample(n)
        log_returns = (model_output * scale) + shift
        log_returns = log_returns.detach().cpu().numpy().flatten()
        return_simulation = np.exp(log_returns[:len])

        price_simulation = np.zeros_like(return_simulation, dtype=self.numpy_dtype)
        price_simulation[0] = init_price

        for i in range(0, price_simulation.shape[0] - 1):
            price_simulation[i + 1] = price_simulation[i] * return_simulation[i]

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
                "price_simulation": wandb.plot.line(
                    price_table,
                    x="timesteps",
                    y="stock price",
                    title=f"{self.symbol} Price Simulation",
                ),
                "return_simulation": wandb.plot.line(
                    return_table,
                    x="timesteps",
                    y="stock price returns",
                    title=f"{self.symbol} Price Return Simulation",
                ),
            }
        )
        return price_simulation, return_simulation

    def min_max_scaling(self, data: torch.Tensor) -> torch.Tensor:
        """Min max scaling of data

        Args:
            data (torch.Tensor): Data to be scaled

        Returns:
            torch.Tensor: scaled data
        """

        self.data_min = torch.min(data)
        data_ = data - self.data_min
        self.data_amplitude = torch.max(data_)

        data_ = data_ / self.data_amplitude

        return data_

    def train_fourier_flow(
        self,
        X: torch.Tensor,
        learning_rate: float,
        gamma: float,
        epochs: int,
        batch_size: int,
        hidden_dim: int,
        num_layer: int,
        device: torch.device,
        run: wr.Run,
    ) -> FourierFlow:
        """Train a Fourier Flow network with given parameters

        Args:
            X (torch.Tensor): Training data
            learning_rate (float): Initial learning rate
            gamma (float): Factor for exponential LRScheduler (in every epoch the new learning rate is lr * gamma)
            epochs (int): Number of epochs to train
            batch_size (int): Batch size.
            hidden_dim (int): Hidden dimensions of Network
            num_layer (int): Number of layers of network
            device (torch.device): Device, CPU or GPU
            run (wr.Run): Wandb run for logging.

        Returns:
            FourierFlow: Trained network ready for sampling.
        """

        T = X.shape[1]

        model = FourierFlow(
            hidden_dim=hidden_dim, T=T, n_layer=num_layer, dtype=self.dtype_str
        )
        # log gradient information
        # run.watch(model, log="all")

        X_scaled = self.min_max_scaling(X)

        model.set_normilizing(X_scaled)

        dataset = TensorDataset(X_scaled)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        # set up model
        model.train()
        model.to(device)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for (batch,) in loader:

                batch = batch.to(device=device)

                optimizer.zero_grad()

                z, log_prob_z, log_jac_det = model(batch)
                loss = torch.mean(-log_prob_z - log_jac_det)

                loss.backward()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()

                epoch_loss += loss.detach().item()

            scheduler.step()
            epoch_loss = epoch_loss / len(loader)
            last_lr = scheduler.get_last_lr()[0]

            run.log({"loss": epoch_loss, "lr": last_lr})

            if epoch % 20 == 0:
                print(
                    f"Epoch: {epoch:>8d},     last loss {epoch_loss:>8.4f},       last learning rate {last_lr:>8.8f}"
                )

        print("Finished training!")

        return model
