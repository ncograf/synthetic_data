import time
from typing import Any, Dict, Tuple

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


class ConditionalFlowGenerator(base_generator.BaseGenerator):
    def fit(
        self,
        price_data: pd.DataFrame,
        config: Dict[str, Any],
        accelerator: Accelerator,
    ):
        """Fit a Conditional Flow Neural Network. I.e. Train the network on the given data.

        Args:
            price_data (pd.DataFrame): Stock marked price datsa
            config(Dict[str, Any]): configuration for training run:
                cond_flow_config:
                    hidden_dim (int): dimension of the hidden layers needs to be even
                    dim (int): dimension of the output / input (equals to the number of stocks to predict)
                    conditional_dim (int): size of the conditional latent representation.
                    n_layer (int): number of spectral layers to be used
                    num_model_layer(int): number of model layer
                    drop_out (float): dropout rate in [0, 1).
                    activation (str): string indicationg the activation function.
                    norm (Literal['layer', 'batch', 'none']): normalization layer to be used.
                    dtype (torch.dtype, optional): type of data. Defaults to torch.float64.
                    dft_scale (float, optional): Amount to scale dft signal. Defaults to 1.
                    dft_shift (float, optional): Amount to shift dft signal. Defaults to 0.
                seq_len : int
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
                init_loc_returns : initial returns for sampling
                fit_score : score of the fit
                symbols : symbol names for model
                network : network name
        """

        if price_data.ndim != 2:
            raise ValueError("Input price data must be two dimensional")

        symbols = price_data.columns

        dtype = config["cond_flow_config"]["dtype"]
        seq_len = config["seq_len"]
        lag = config["lag"]

        # transform dataset for training
        price_data = price_data.astype(TypeConverter.str_to_numpy(dtype))
        data = torch.from_numpy(price_data.values)
        data = data[~torch.any(data.isnan(), dim=1)]  # remove nan data
        zero_price = data[0]
        log_returns = torch.log(data[1:] / (data[:-1] + 1e-8))

        # split the data
        X = np.lib.stride_tricks.sliding_window_view(
            log_returns, seq_len, axis=0, writeable=False
        )
        X = torch.tensor(X[::lag])
        X = X.swapaxes(1, 2)

        # train model
        model, shift, scale, last_epoch_loss = self.train_cond_flow(
            accelerator=accelerator, X=X, config=config
        )

        # save model after training DO NOT CHANGE THE NAME as we need it when downloading the models
        model_dict = {
            "state_dict": model.state_dict(),
            "init_params": model.get_model_info(),
            "network": str(model),
            "scale": scale,
            "shift": shift,
            "init_price": zero_price,
            "init_log_returns": log_returns[
                :seq_len
            ],  # take the first rows of the training data
            "fit_score": last_epoch_loss,
            "symbols": symbols,
        }

        accelerator.free_memory()
        return model_dict

    def sample(
        self, length: int, burn: int, config: Dict[str, any]
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generate data from the trained model

        If model_version = None, cosider to disable wandb with
        `os.environ['WANDB_MODE'] = 'disabled'`

        Args:
            length (int): lengh of generated seqence.
            burn (int): ignored.
            config (Dict[str, Any]): sample model configuration:
                state_dict : torch model weights
                init_params : model init params
                scale : data scaling
                shift : data shift
                init_price : initial price for sampling
                init_log_returns : preceeding condition for continuing the sequence

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: price data and return data
        """
        dtype = config["dtype"] if "dtype" in config else "float64"
        scale = config["scale"]
        shift = config["shift"]
        init_price = config["init_price"]
        init_log_returns = config["init_log_returns"]

        model = ConditionalFlow(**config["init_params"])
        model.load_state_dict(config["state_dict"])
        model.dtype = TypeConverter.str_to_torch(dtype)

        init_samples = (init_log_returns - shift) / scale
        model_output = model.sample(n=length + burn, x=init_samples)
        log_returns = (model_output * scale) + shift
        log_returns = log_returns.detach().cpu().numpy()
        return_simulation = np.exp(log_returns[:length])

        price_simulation = np.zeros(
            (return_simulation.shape[0] + 1, return_simulation.shape[1]),
            dtype=TypeConverter.str_to_numpy(dtype),
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
        data_ = data - shift
        scale = torch.max(data_)

        data_ = data_ / scale

        return data_, shift, scale

    def train_cond_flow(
        self,
        X: torch.Tensor,
        config: Dict[str, Any],
        accelerator: Accelerator,
    ) -> Tuple[ConditionalFlow, float, float, float]:
        """Train a Conditional Flow network with given parameters

        Args:
            X (torch.Tensor): Training data
            config(Dict[str, Any]): configuration for training run:
                cond_flow_config:
                    hidden_dim : int
                    seq_len : int
                    n_layer : int
                    output_dim : int
                    dtype: str
                batch_size : int
                epochs : int
                optim_config: config for adam optimizer (e.g. lr : float)
            accelerator (Accelerator): For fast training

        Returns:
            Tuple[ConditionalFlow, float, float, flaot]: Trained network ready for sampling, shift, scale, last epoch loss
        """

        batch_size = config["batch_size"]
        epochs = config["epochs"]

        model_ff = ConditionalFlow(**config["cond_flow_config"])
        # log gradient information
        # run.watch(model, log="all")

        X_scaled, shift, scale = self.min_max_scaling(X)
        model_ff.set_normilizing(X_scaled)

        dataset = TensorDataset(X_scaled)
        loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

        optimizer = torch.optim.Adam(model_ff.parameters(), **config["optim_config"])
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

            if wandb.run is not None:
                wandb.log(
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
                    f"epoch: {epoch + 1:>8d}/{epochs},\tlast loss {epoch_loss:>8.4f},\tlast learning rate {last_lr:>8.8f}tests/test_fourier_flow_generator.py tests/test_fourier_flows.py tests/test_fourier_transform_layer.py"
                )

        print("Finished training!")

        return accelerator.unwrap_model(model), shift, scale, epoch_loss
