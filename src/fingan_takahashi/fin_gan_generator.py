import time
from typing import Any, Dict, Tuple

import base_generator
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from accelerate import Accelerator
from fin_gan import FinGan
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from type_converter import TypeConverter

import wandb


class FinGanGenerator(base_generator.BaseGenerator):
    def fit(
        self,
        price_data: pd.DataFrame,
        config: Dict[str, Any],
        accelerator: Accelerator,
        sym: str = "",
    ):
        """Fit a Fourier Flow Neural Network. I.e. Train the network on the given data.

        Args:
            price_data (pd.DataFrame): Stock marked price datsa
            config (Dict[str, Any]): configuration for training run:
                fingan_config:
                    input_dim: input dimension for generator
                    arch : generator base architectur -> MLP
                    activation : generator activation function
                    drop_out : generator dropout
                    layers : sizes of hidden layers
                    norm : norm applied between layers
                seq_len: int
                dtype: str
                epochs: int
                batch_size: int
                optim_gen_config: config for adam optimizer (e.g. lr : float)
                optim_disc_config: config for adam optimizer (e.g. lr : float)
                lr_config : config for exponential lr_scheduler (e.g. gamma : float)
            accelerator (Accelerator): For fast training
            sym (str, optional) : symbol for logging. Defaults to ""

        Returns:
            Dict[str, Any]: Model description ready for sampling
                state_dict : torch model weights
                init_params : model init params
                scale : data scaling
                shift : data shift
                init_price : initial price for sampling
                symbol : symbol name for model
                network : network name
        """

        if price_data.ndim != 1:
            raise ValueError("Input price data must be one dimensional")

        seq_len = config["seq_len"]
        dtype = TypeConverter.str_to_torch(config["dtype"])

        # transform dataset for training
        data = torch.from_numpy(price_data.values)
        data_mask = ~torch.isnan(data)
        data = data[data_mask]
        _zero_price = data[0]
        log_returns = torch.log(data[1:] / data[:-1])

        # periodically enlarge sample space
        n_ret = log_returns.shape[0]
        while n_ret < 2 * seq_len:
            log_returns = torch.cat(
                [log_returns, log_returns[: (2 * seq_len - n_ret)]], dim=0
            )
            n_ret = log_returns.shape[0]

        # split the data
        X = np.lib.stride_tricks.sliding_window_view(
            log_returns, seq_len, axis=0, writeable=False
        )
        X = torch.tensor(X, dtype=dtype)
        X, shift, scale = self.min_max_scaling(X)

        # train model
        _model, score = self.train_fin_gan(
            accelerator=accelerator,
            X=X,
            config=config,
            sym=sym,
        )

        # save model after training DO NOT CHANGE THE NAME as we need it when downloading the models
        model_dict = {
            "state_dict": _model.state_dict(),
            "init_params": _model.get_model_info(),
            "network": str(_model),
            "scale": scale,
            "shift": shift,
            "init_price": _zero_price,
            "fit_score": score,
            "symbol": sym,
        }

        return model_dict

    def sample(
        self, length: int, burn: int, config: Dict[str, any]
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Generate data from the trained model

        If model_version = None, cosider to disable wandb with
        `os.environ['WANDB_MODE'] = 'disabled'`

        Args:
            length (int): lengh of generated seqence.
            burn (int): number of elemnts to burn before starting generate output.
            config (Dict[str, Any]): sample model configuration:
                state_dict : torch model weights
                init_params : model init params
                scale : data scaling
                shift : data shift
                init_price : initial price for sampling

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: price data and return data
        """

        dtype = config["dtype"] if "dtype" in config else "float32"
        scale = config["scale"]
        shift = config["shift"]
        init_price = config["init_price"]

        model = FinGan(**config["init_params"])
        model.load_state_dict(config["state_dict"])

        num_seq = length // (config["init_params"]["gen_config"]["output_dim"] - 1) + 1
        model_output = model.sample(num_seq, burn=burn)
        model_output = model_output.flatten()[:length]
        log_returns = (model_output * scale) + shift
        log_returns = log_returns.detach().cpu().numpy().flatten()
        return_simulation = np.exp(log_returns)

        price_simulation = np.zeros(
            (return_simulation.shape[0] + 1), dtype=TypeConverter.str_to_numpy(dtype)
        )
        price_simulation[0] = init_price

        for i in range(0, price_simulation.shape[0] - 1):
            price_simulation[i + 1] = price_simulation[i] * return_simulation[i]

        return price_simulation, return_simulation

    def min_max_scaling(self, data: torch.Tensor) -> torch.Tensor:
        """Min max scaling of data

        Args:
            data (torch.Tensor): Data to be scaled

        Returns:
            Tuple[torch.Tensor, float, float]: scaled data, shift, scale
        """

        shift = torch.min(data)
        data_ = data - shift
        scale = torch.max(data_)

        data_ = data_ / scale

        return data_, shift, scale

    def train_fin_gan(
        self,
        X: torch.Tensor,
        config: Dict[str, Any],
        accelerator: Accelerator,
        sym: str = "",
    ) -> FinGan:
        """Train a Time Gan network with given parameters

        Args:
            X (torch.Tensor): Training data
            config (Dict[str, Any]): configuration for training run:
                fingan_config:
                    input_dim: input dimension for generator
                    arch : generator base architectur -> MLP
                    activation : generator activation function
                    drop_out : generator dropout
                    layers : size of hidden layers
                    norm : norm applied between layers
                seq_len: int
                dtype: str
                epochs: int
                batch_size: int
                optim_gen_config: config for adam optimizer (e.g. lr : float)
                optim_disc_config: config for adam optimizer (e.g. lr : float)
                lr_config : config for exponential lr_scheduler (e.g. gamma : float)
            accelerator (Accelerator): For fast training
            sym (str, optional) : symbol for logging. Defaults to ""

        Returns:
            Tuple[TimeGan, float]: Trained network ready for sampling, final sum of generator and discriminator loss
        """

        gen_config = config["fingan_config"]
        gen_config["output_dim"] = config["seq_len"]

        disc_config = {
            "arch": "fin_gan_disc",
            "input_dim": config["seq_len"],
            "output_dim": 1,
            "activation": {"leaky_relu": 0.2},
            "drop_out": 0.5,
        }

        model = FinGan(
            gen_config=gen_config, disc_config=disc_config, dtype=config["dtype"]
        )

        batch_size = config["batch_size"]
        epochs = config["epochs"]

        y_real = torch.rand(size=(X.shape[0],)) * 0.2 + 0.1
        y_fake = torch.rand(size=(X.shape[0],)) * 0.2 + 0.9
        y = torch.stack([y_real, y_fake], dim=1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

        gen_optim = torch.optim.Adam(
            model.gen.parameters(), **config["optim_gen_config"]
        )
        disc_optim = torch.optim.Adam(
            model.disc.parameters(), **config["optim_disc_config"]
        )

        accelerator.free_memory()
        loader, model, gen_optim, disc_optim = accelerator.prepare(
            loader, model, gen_optim, disc_optim
        )

        cross_entropy_loss = CrossEntropyLoss()

        # Bad configuration might make the model collaps
        assert model is not None

        # train reconstruction network part
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_time = time.time()

            for real_series, y in loader:
                b_size = real_series.shape[0]

                gen_optim.zero_grad()
                disc_optim.zero_grad()

                gen_series = model.sample(batch_size=b_size)
                real_series = torch.nan_to_num(real_series)

                all_series = torch.unsqueeze(
                    torch.cat([real_series, gen_series], dim=0), dim=1
                )

                disc_y = model.disc(all_series)

                # y will contain labels for real and synthetic data already
                y = torch.cat([y[:, 0], y[:, 1]], dim=0)

                loss = cross_entropy_loss(disc_y.flatten(), y)
                epoch_loss += loss.item()

                accelerator.backward(loss)

                gen_optim.step()
                disc_optim.step()

                epoch_loss = loss.item()

                if wandb.run is not None:
                    wandb.log(
                        {
                            f"loss/{sym}": epoch_loss,
                            f"epoch_time/{sym}": time.time() - epoch_time,
                            "epoch": epoch,
                        }
                    )

            n = 1
            if epoch % n == n - 1:
                print(
                    f"epoch/{sym}: {epoch + 1:>8d}/{epochs},\tlast loss {epoch_loss:>8.4f}"
                )

        print(f"Finished training {sym}!")

        accelerator.free_memory()
        return accelerator.unwrap_model(model), epoch_loss
