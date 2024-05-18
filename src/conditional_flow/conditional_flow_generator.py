import time
from pathlib import Path
from typing import Tuple

import base_generator
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from accelerate import Accelerator
from conditional_flow import ConditionalFlow
from torch.utils.data import DataLoader, TensorDataset
from type_converter import TypeConverter

import wandb
from wandb.sdk import wandb_run as wr


class ConditionalFlowGenerator(base_generator.BaseGenerator):
    def __init__(
        self,
        name: str = "conditional_flow",
        dtype: str = "float32",
        cache: str | Path = "data/cache",
    ):
        """Initialize Conditinal Flow Generator

        Args:
            symbols (List[str]) : Names of the symbols to be created / trained on.
            name (str, optional): Generator name. Defaults to 'conditional_flow'.
            dtype (str, optional): dtype of model. Defaults to 'float32'.
            cache (str | Path, optional): cache location to store model and artifacts. Defaults to 'data/cache'.
        """
        base_generator.BaseGenerator.__init__(self, name)
        self.shift = 0
        self.scale = 1

        if not isinstance(dtype, str):
            self.dtype_str = TypeConverter.type_to_str(dtype)
            self.torch_dtype = TypeConverter.str_to_torch(self.dtype_str)
            self.numpy_dtype = TypeConverter.str_to_numpy(self.dtype_str)
        else:
            self.dtype_str = TypeConverter.extract_dtype(dtype)
            self.torch_dtype = TypeConverter.str_to_torch(self.dtype_str)
            self.numpy_dtype = TypeConverter.str_to_numpy(self.dtype_str)

        self._model: torch.Module | None = None
        self.cache = Path(cache)
        if not self.cache.exists():
            raise ValueError(f"Cache {str(self.cache)} does not exist.")

    def fit_model(
        self,
        run: wr.Run,
        accelerator: Accelerator,
        price_data: pd.DataFrame,
        learning_rate: float = 0.001,
        gamma: float = 0.995,
        epochs: int = 100,
        batch_size: int = 512,
        hidden_dim: int = 64,
        num_layer: int = 4,
        lag: int = 1,
        seq_len: int = 512,
    ):
        """Fit a Conditional Flow Neural Network. I.e. Train the network on the given data.

        Args:
            price_data (pd.DataFrame): Stock marked price data TxD
            learning_rate (float, optional): Initial learing rate. Defaults to 0.001.
            gamma (float, optional): Exponential decay base of learning rate. Defaults to 0.995.
            epochs (int, optional): Number of training epochs. Defaults to 1000.
            batch_size (int, optional): Size of the batches to be processed. Defaults to 128.
            hidden_dim (int, optional): Hidden dimensions of the model recommended around 2 * seq_len. Defaults to 200.
            num_layer (int, optional): Number of layers in the Network. Defaults to 10.
            lag (int, optional): lag between the indivisual seqences when splitting the price. Defaults to 1.
            seq_len (int, optional): seqence lenght. Defaults to 512.
            notes (str, optional): notes describing the task. Defaults to "".
            tags (List[str], optional): List of tags. Defaults to [].

        Raises:
            ValueError: If the input is no a single price seqence
        """

        if price_data.ndim != 2:
            raise ValueError("Input price data must be two dimensional")

        config = {
            "hidden_dim": hidden_dim,
            "num_layer": num_layer,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "gamma": gamma,
        }

        symbols = price_data.columns

        # transform dataset for training
        price_data = price_data.astype(self.numpy_dtype)
        data = torch.from_numpy(price_data.values)
        data = data[~torch.any(data.isnan(), dim=1)]  # remove nan data
        self._zero_price = data[0]
        log_returns = torch.log(data[1:] / (data[:-1] + 1e-8))

        # split the data
        X = np.lib.stride_tricks.sliding_window_view(
            log_returns, seq_len, axis=0, writeable=False
        )
        X = torch.tensor(X[::lag])
        X = X.swapaxes(1, 2)

        # train model
        self._model = self.train_cond_flow(
            accelerator=accelerator, run=run, X=X, **config
        )

        # save model after training DO NOT CHANGE THE NAME as we need it when downloading the models
        model_path = self.cache / "conditional_flow.pth"
        model_dict = {
            "state_dict": self._model.state_dict(),
            "init_params": self._model.get_model_info(),
            "network": str(self._model),
            "scale": self.scale,
            "shift": self.shift,
            "init_price": self._zero_price,
            "symbols": symbols,
        }

        torch.save(model_dict, model_path)  # will be compressed automatically
        model_artifact = wandb.Artifact(
            name="conditional_flow_model",
            type="model",
            metadata={
                "Info": "This information describes the features stored in the artifact",
                "state_dict": "Model weights to be loaded into a torch Module",
                "init_params": "Parameters to initialize network",
                "network": "Network class used in Training",
                "scale": "Scaling of the training data X_train = (X - shift) / scale",
                "shift": "Shift of the training data X_train = (X - shift) / scale",
                "init_price": "Initial Price for the stock for reconstruction",
                "symbols": "Symbols on which the model was fitted",
            },
        )
        model_artifact.add_file(model_path)
        run.log_artifact(model_artifact)

    def model(self) -> ConditionalFlow:
        return self._model

    def check_model(self):
        if self._model is None:
            raise RuntimeError("Model must bet set before.")

    def generate_data(
        self,
        model: ConditionalFlow,
        x: torch.Tensor,
        scale: float,
        shift: float,
        init_price: float,
        len_: int = 500,
        burn: int = 100,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generate data from the trained model

        If model_version = None, cosider to disable wandb with
        `os.environ['WANDB_MODE'] = 'disabled'`

        Args:
            model (ConditionalFlow): model to sample from.
            x (torch.Tensor): initial prices for all stocks.
            len (int, optional): lengh of generated seqence. Defaults to 500.
            burn (int, optional): ignored here. Defaults to 100.
            tags (List[str], optional): list of tags for wandb. Defaults to [].
            notes (str, optional): Notes in markdown. Defaults to "".

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: price data and return data
        """

        x = x * self.scale + self.shift
        model_output = model.sample(len_ + burn, x)[burn:]
        log_returns = (model_output * scale) + shift
        log_returns = log_returns.detach().cpu().numpy()
        return_simulation = np.exp(log_returns[:len_])

        price_simulation = np.zeros_like(return_simulation, dtype=self.numpy_dtype)
        price_simulation[0] = init_price

        for i in range(0, price_simulation.shape[0] - 1):
            price_simulation[i + 1] = price_simulation[i] * return_simulation[i]

        return price_simulation, return_simulation

    def min_max_scaling(self, data: torch.Tensor) -> torch.Tensor:
        """Min max scaling of data

        Args:
            data (torch.Tensor): Data to be scaled

        Returns:
            torch.Tensor: scaled data
        """

        self.shift = torch.min(data)
        data_ = data - self.shift
        self.scale = torch.max(data_)

        data_ = data_ / self.scale

        return data_

    def train_cond_flow(
        self,
        accelerator: Accelerator,
        run: wr.Run,
        X: torch.Tensor,
        learning_rate: float,
        gamma: float,
        epochs: int,
        batch_size: int,
        hidden_dim: int,
        num_layer: int,
    ) -> ConditionalFlow:
        """Train a Conditional Flow network with given parameters

        Args:
            accelerator (Accelerator): accelerator to be used for speedup,
            run (Run): Wandb run for logging,
            X (torch.Tensor): Training data
            learning_rate (float): Initial learning rate
            gamma (float): Factor for exponential LRScheduler (in every epoch the new learning rate is lr * gamma)
            epochs (int): Number of epochs to train
            batch_size (int): Batch size.
            hidden_dim (int): Hidden dimensions of Network
            num_layer (int): Number of layers of network

        Returns:
            ConditionalFlow: Trained network ready for sampling.
        """

        D = X.shape[2]

        model_ff = ConditionalFlow(
            hidden_dim=hidden_dim, output_dim=D, n_layer=num_layer, dtype=self.dtype_str
        )
        # log gradient information
        # run.watch(model, log="all")

        X_scaled = self.min_max_scaling(X)
        model_ff.set_normilizing(X_scaled)

        dataset = TensorDataset(X_scaled)
        loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

        optimizer = torch.optim.Adam(model_ff.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        loader, model, optimizer, scheduler = accelerator.prepare(
            loader, model_ff, optimizer, scheduler
        )

        # Bad configuration might make the model collaps
        assert model is not None

        # set up model
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_time = time.time()

            for (batch,) in loader:
                optimizer.zero_grad()

                _, log_prob_z, log_jac_det = model(batch)
                loss = torch.mean(-log_prob_z - log_jac_det)

                accelerator.backward(loss)

                optimizer.step()

                epoch_loss += loss.detach().item()

            scheduler.step()
            epoch_loss = epoch_loss / len(loader)
            last_lr = scheduler.get_last_lr()[0]

            run.log(
                {
                    "loss": epoch_loss,
                    "lr": last_lr,
                    "epoch_time": time.time() - epoch_time,
                    "epoch": epoch,
                }
            )

            n = 20
            if epoch % n == n - 1:
                print(
                    f"epoch: {epoch + 1:>8d}/{epochs},\tlast loss {epoch_loss:>8.4f},\tlast learning rate {last_lr:>8.8f}"
                )

        print("Finished training!")

        return accelerator.unwrap_model(model)
