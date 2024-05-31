from typing import Any, Dict, Tuple

import base_generator
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from accelerate import Accelerator
from fourier_flow import FourierFlow
from torch.utils.data import DataLoader, TensorDataset
from type_converter import TypeConverter as TC

import wandb


class FourierFlowGenerator(base_generator.BaseGenerator):
    def fit(
        self,
        price_data: pd.DataFrame,
        config: Dict[str, Any],
        accelerator: Accelerator,
        sym: str = "",
    ) -> Dict[str, Any]:
        """Fit a Fourier Flow Neural Network. I.e. Train the network on the given data.

        Args:
            price_data (pd.DataFrame): Stock marked price datsa
            config(Dict[str, Any]): configuration for training run:
                fourier_flow_config:
                    hidden_dim : int
                    seq_len : int
                    num_layer : int
                dtype: str
                epochs: int
                batch_size : int
                lag : int
                optim_config: config for adam optimizer (e.g. lr : float)
                lr_config : config for exponential lr_scheduler (e.g. gamma : float)
            accelerator (Accelerator): accelerator for speedup
            sym (str): symbol of the fitted price data

        Raises:
            ValueError: If the input is no a single price seqence

        Returns:
            Dict[str, Any]: Model description ready for sampling
                state_dict : torch model weights
                init_params : model init params
                scale : data scaling
                shift : data shift
                init_price : initial price for sampling
                fit_score : score of the fit
                symbol : symbol name for model
                network : network name
        """

        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame()

        if len(price_data.columns) != 1:
            raise ValueError("Input price data must be one dimensional")

        dtype = config["dtype"]
        seq_len = config["fourier_flow_config"]["seq_len"]
        lag = config["lag"]

        # transform dataset for training
        price_data = price_data.astype(TC.str_to_numpy(dtype))
        data = torch.from_numpy(price_data.values)
        data_mask = ~torch.isnan(data)
        data = data[data_mask]
        zero_price = data[0]
        log_returns = torch.log(data[1:] / data[:-1])

        # split the data
        X = np.lib.stride_tricks.sliding_window_view(
            log_returns, seq_len, axis=0, writeable=False
        )
        X = torch.tensor(X[::lag, :])

        # train model
        model, shift, scale, last_epoch_loss = self.train_fourier_flow(
            accelerator=accelerator, X=X, config=config, sym=sym
        )

        # save model after training DO NOT CHANGE THE NAME as we need it when downloading the models
        model_dict = {
            "state_dict": model.state_dict(),
            "init_params": model.get_model_info(),
            "network": str(model),
            "scale": scale,
            "shift": shift,
            "init_price": zero_price,
            "fit_score": last_epoch_loss,
            "symbol": sym,
        }

        accelerator.free_memory()
        return model_dict

    def sample(
        self, length: int, burn: int, config: Dict[str, any]
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generate data from the trained model

        Args:
            length (int): lengh of generated seqence.
            burn (int): ignored.
            config (Dict[str, Any]): sample model configuration:
                state_dict : torch model weights
                init_params : model init params
                scale : data scaling
                shift : data shift
                init_price : initial price for sampling

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: price data and return data
        """

        dtype = config["dtype"] if "dtype" in config else "float64"
        scale = config["scale"]
        shift = config["shift"]
        init_price = config["init_price"]

        model = FourierFlow(**config["init_params"])
        model.load_state_dict(config["state_dict"])
        model.dtype = TC.str_to_torch(dtype)

        n = length // model.seq_len + 1
        model_output = model.sample(n)
        log_returns = (model_output * scale) + shift
        log_returns = log_returns.detach().cpu().numpy().flatten()
        return_simulation = np.exp(log_returns[:length])

        price_simulation = np.zeros(
            (return_simulation.shape[0] + 1), dtype=TC.str_to_numpy(dtype)
        )
        price_simulation[0] = init_price

        for i in range(0, price_simulation.shape[0] - 1):
            price_simulation[i + 1] = price_simulation[i] * return_simulation[i]

        return price_simulation, return_simulation

    def min_max_scaling(self, data: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Min max scaling of data

        Args:
            data (torch.Tensor): Data to be scaled

        Returns:
            Tuple[tensor, float, float]: scaled data, shift, scale
        """

        shift = torch.min(data)
        data = data - shift
        scale = torch.max(data)

        data = data / scale

        return data, shift, scale

    def train_fourier_flow(
        self,
        X: torch.Tensor,
        config: Dict[str, Any],
        accelerator: Accelerator,
        sym: str = "",
    ) -> Tuple[FourierFlow, float, float, float]:
        """Train a Fourier Flow network with given parameters

        Args:
            X (torch.Tensor): Training data
            config(Dict[str, Any]): configuration for training run:
                fourier_flow_config:
                    hidden_dim : int
                    seq_len : int
                    num_layer : int
                    num_model_layer: int
                    arch : 'MLP or 'LSTM'
                    bidirect : bool
                dtype: str
                batch_size : int
                epochs : int
                optim_config: config for adam optimizer (e.g. lr : float)
                lr_config : config for exponential lr_scheduler (e.g. gamma : float)
            accelerator (Accelerator): For fast training
            sym (str, optional) : symbol for logging. Defaults to ""

        Returns:
            Tuple[FourierFlow, float, float, float]: Trained network ready for sampling, shift, scale, last_loss
        """

        dtype = TC.str_to_torch(config["dtype"])
        batch_size = config["batch_size"]
        epochs = config["epochs"]
        model_ff = FourierFlow(**config["fourier_flow_config"], dtype=dtype)
        # log gradient information
        # run.watch(model, log="all")

        X_scaled, shift, scale = self.min_max_scaling(X)
        model_ff.set_normilizing(X_scaled)  # sets additional FFT scaling

        dataset = TensorDataset(X_scaled)
        loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

        optimizer = torch.optim.Adam(model_ff.parameters(), **config["optim_config"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, **config["lr_config"]
        )

        loader, model, optimizer, scheduler = accelerator.prepare(
            loader, model_ff, optimizer, scheduler
        )

        # set up model
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
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

            if wandb.run is not None:
                wandb.log(
                    {
                        f"loss/{sym}": epoch_loss,
                        f"lr/{sym}": last_lr,
                        "epoch": epoch,
                    }
                )

            n = 20
            if epoch % n == n - 1:
                print(
                    f"{sym}-epoch: {epoch + 1:>8d}/{epochs},\tlast loss {epoch_loss:>8.4f},\tlast learning rate {last_lr:>8.8f}"
                )

        print(f"Finished training {sym}!")

        return accelerator.unwrap_model(model), shift, scale, epoch_loss
