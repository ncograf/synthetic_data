import time
from pathlib import Path
from typing import Tuple

import base_generator
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from accelerate import Accelerator
from time_gan import TimeGan
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from type_converter import TypeConverter

import wandb
from wandb.sdk import wandb_run as wr


class TimeGanGenerator(base_generator.BaseGenerator):
    def __init__(
        self,
        symbol: str = "NoName",
        name: str = "TimeGan",
        dtype: str = "float32",
        cache: str | Path = "data/cache",
    ):
        """Initialize Time Gan Generator

        Args:
            symbol (str, optional) : Name of the symbol to be created. Defaults to 'NoName'.
            name (str, optional): Generator name. Defaults to "TimeGan".
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
        self._symbol = symbol

    @property
    def symbol(self) -> str:
        return self._symbol

    @symbol.setter
    def symbol(self, value: str):
        self._symbol = value

    def fit_model(
        self,
        run: wr.Run,
        accelerator: Accelerator,
        price_data: pd.DataFrame,
        learning_rate: float = 0.001,
        gamma: float = 0.995,
        epochs: int = 1000,
        batch_size: int = 128,
        hidden_dim: int = 200,
        num_layer: int = 10,
        lag: int = 1,
        seq_len: int = 101,
        embed_dim: int = 2,
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

        config = {
            "hidden_dim": hidden_dim,
            "num_layer": num_layer,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "embed_dim": embed_dim,
        }

        # transform dataset for training
        data = torch.from_numpy(price_data.values)
        data_mask = ~torch.isnan(data)
        data = data[data_mask]
        self._zero_price = data[0]
        log_returns = torch.log(data[1:] / data[:-1])

        # split the data
        X = np.lib.stride_tricks.sliding_window_view(
            log_returns, seq_len, axis=0, writeable=False
        )
        X = torch.tensor(X[::lag], dtype=self.torch_dtype)
        X = X.unsqueeze(dim=-1)

        # train model
        self._model = self.train_time_gan(
            accelerator=accelerator, run=run, X=X, **config
        )

        # save model after training DO NOT CHANGE THE NAME as we need it when downloading the models
        model_path = self.cache / f"{self.symbol}.pth"
        model_dict = {
            "state_dict": self._model.state_dict(),
            "init_params": self._model.get_model_info(),
            "network": str(self._model),
            "scale": self.scale,
            "shift": self.shift,
            "init_price": self._zero_price,
            "symbol": self.symbol,
        }

        torch.save(model_dict, model_path)  # will be compressed automatically
        model_artifact = wandb.Artifact(
            name=f"time_gan_model_{self.symbol}",
            type="model",
            metadata={
                "symbol": self.symbol,
                "Info": "This information describes the features stored in the artifact",
                "state_dict": "Model weights to be loaded into a torch Module",
                "init_params": "Parameters to initialize network",
                "network": "Network class used in Training",
                "scale": "Scaling of the training data X_train = (X - shift) / scale",
                "shift": "Shift of the training data X_train = (X - shift) / scale",
                "init_price": "Initial Price for the stock for reconstruction",
                "symbol_": "The symbol indicates for what symbol(s) the model was fitted",
            },
        )
        model_artifact.add_file(model_path)
        run.log_artifact(model_artifact)

    def model(self) -> TimeGan:
        return self._model

    def check_model(self):
        if self._model is None:
            raise RuntimeError("Model must bet set before.")

    def generate_data(
        self,
        model: TimeGan,
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
            model (FourierFlow): model to sample from.
            len (int, optional): lengh of generated seqence. Defaults to 500.
            burn (int, optional): ignored here. Defaults to 100.
            tags (List[str], optional): list of tags for wandb. Defaults to [].
            notes (str, optional): Notes in markdown. Defaults to "".

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: price data and return data
        """

        n = len_ // model.T + 1

        model_output = model.sample(n, bunr=burn)
        log_returns = (model_output * scale) + shift
        log_returns = log_returns.detach().cpu().numpy().flatten()
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

    def train_time_gan(
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
        embed_dim: int,
    ) -> TimeGan:
        """Train a Time Gan network with given parameters

        Args:
            accelerator (Accelerator): Accelerator used for the training
            run (wr.Run): Wandb run for logging.
            X (torch.Tensor): Training data
            learning_rate (float): Initial learning rate
            gamma (float): Factor for exponential LRScheduler (in every epoch the new learning rate is lr * gamma)
            epochs (int): Number of epochs to train
            batch_size (int): Batch size.
            hidden_dim (int): Hidden dimensions of Network
            num_layer (int): Number of layers of network
            embed_dim (int): Embedding dimensions

        Returns:
            TimeGan: Trained network ready for sampling.
        """

        model_tg = TimeGan(
            hidden_dim=hidden_dim, embed_dim=embed_dim, num_layer=num_layer, n_stocks=1
        )
        # log gradient information
        # run.watch(model, log="all")

        X_scaled = self.min_max_scaling(X)

        dataset = TensorDataset(X_scaled)
        loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

        # Define optimizer
        recon_params = list(model_tg.embedder.parameters()) + list(
            model_tg.recovery.parameters()
        )
        opt_recon = torch.optim.Adam(recon_params, lr=learning_rate)

        super_params = list(model_tg.supervisor.parameters())
        opt_super = torch.optim.Adam(super_params, lr=learning_rate)

        gen_params = list(model_tg.generator.parameters()) + recon_params + super_params
        opt_gener = torch.optim.Adam(gen_params, lr=learning_rate)

        discr_params = list(model_tg.discriminator.parameters())
        opt_discr = torch.optim.Adam(discr_params, lr=learning_rate)

        accelerator.free_memory()
        loader, model, opt_recon, opt_super, opt_gener, opt_discr = accelerator.prepare(
            loader, model_tg, opt_recon, opt_super, opt_gener, opt_discr
        )

        print("models are loaded")

        cross_entropy_loss = CrossEntropyLoss()

        # Bad configuration might make the model collaps
        assert model is not None

        # train reconstruction network part
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_time = time.time()

            print("start first epoch")

            for (x,) in loader:
                opt_recon.zero_grad()

                h = model.embed(x)
                x_tilde = model.recover(h)

                loss = torch.mean(
                    (x_tilde.flatten() - x.flatten()) ** 2
                )  # reconstructed x and real x

                accelerator.backward(loss)
                opt_recon.step()

                epoch_loss += loss.item()

            run.log(
                {
                    f"recon_loss/{self.symbol}": epoch_loss,
                    # f"reconstruction_lr/{self.symbol}": last_lr,
                    f"recon_epoch_time/{self.symbol}": time.time() - epoch_time,
                    "epoch": epoch,
                }
            )

            print(f"epoch took {time.time() - epoch_time}")

        # train with supervised lossx.shape + (1)
        # note that this is a hack needed to replace teacher forcing
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_time = time.time()

            for (x,) in loader:
                opt_super.zero_grad()

                h = model.embed(x)
                h_super = model.predict_next(h)
                loss = torch.mean((h[:, :-1] - h_super[:, 1:]) ** 2)  # supervisor loss

                accelerator.backward(loss)
                opt_super.step()
                # scd_super.step()

                epoch_loss += loss.item()

            run.log(
                {
                    f"super_loss/{self.symbol}": epoch_loss,
                    # f"reconstruction_lr/{self.symbol}": last_lr,
                    f"super_epoch_time/{self.symbol}": time.time() - epoch_time,
                    "epoch": epoch,
                }
            )

            print(f"epoch took {time.time() - epoch_time}")

        # GAN training
        for epoch in range(epochs):
            epoch_gen_loss = 0
            epoch_super_loss = 0
            epoch_time = time.time()
            n_gen = 3

            # Train the generator twice
            for _ in range(n_gen):
                for (x,) in loader:
                    opt_gener.zero_grad()

                    h_real = model.embed(x)
                    h_gen = model.generate_embed(*x.shape[:2])

                    h_hat = model.predict_next(
                        h_gen
                    )  # generator and supervisor are used to enable teacher forcing
                    x_hat = model.recover(h_hat)  # get back the x from the generator

                    h_hat_super = model.predict_next(h_real)
                    x_hat_real = model.recover(
                        h_real
                    )  # get back the x from the supervisor

                    y_fake = model.discriminate_embed(h_hat)
                    y_fake_gen = model.discriminate_embed(h_gen)

                    # embedder and supervised loss
                    loss_supervisor = torch.mean(
                        (h_real[:, :-1] - h_hat_super[:, 1:]) ** 2
                    )  # supervisor loss
                    loss_embedder = torch.mean(
                        (x.flatten() - x_hat_real.flatten()) ** 2
                    )  # embedder loss
                    loss_emb_sup = (
                        10 * torch.sqrt(loss_embedder) + 0.1 * loss_supervisor
                    )

                    # moment losses not mentioned in the paper but in the code
                    loss_std = torch.mean(
                        torch.abs(torch.std(x_hat, dim=0) - torch.std(x, dim=0))
                    )
                    loss_mu = torch.mean(
                        torch.abs(torch.mean(x_hat, dim=0) - torch.mean(x, dim=0))
                    )
                    loss_moments = loss_mu + loss_std

                    # generator losses
                    loss_fake_gen = cross_entropy_loss(
                        torch.ones_like(y_fake_gen), y_fake_gen
                    )
                    loss_fake = cross_entropy_loss(torch.ones_like(y_fake), y_fake)
                    loss_generator = (
                        loss_fake
                        + gamma * loss_fake_gen
                        + 100 * torch.sqrt(loss_supervisor)
                        + 100 * loss_moments
                    )

                    # compute gradients
                    loss_all = loss_emb_sup + loss_generator

                    accelerator.backward(loss_all)

                    opt_gener.step()
                    # scd_gener.step()

                    epoch_gen_loss += loss_generator.item()
                    epoch_super_loss += loss_emb_sup.item()

            epoch_gen_loss /= n_gen
            epoch_super_loss /= n_gen

            run.log(
                {
                    f"generator_loss/{self.symbol}": epoch_gen_loss,
                    f"embed_loss/{self.symbol}": epoch_gen_loss,
                    # f"reconstruction_lr/{self.symbol}": last_lr,
                    f"generator_epoch_time/{self.symbol}": time.time() - epoch_time,
                    "epoch": epoch,
                }
            )

            # start discriminator training
            epoch_disc_loss = 0
            epoch_time = time.time()

            for (x,) in loader:
                opt_discr.zero_grad()

                h_gen = model.generate_embed(*x.shape[:2])
                h_real = model.embed(x)
                h_gen = model.generate_embed(*x.shape[:2])
                h_hat = model.predict_next(
                    h_gen
                )  # generator and supervisor are used to enable teacher forcing

                y_fake = model.discriminate_embed(h_hat)
                y_fake_gen = model.discriminate_embed(h_gen)
                y_real = model.discriminate_embed(
                    h_real
                )  # I think these outputs are shifted back by one as they are not fed throught the supervisor

                # discriminator loss
                loss_disc_fake_gen = cross_entropy_loss(
                    torch.zeros_like(y_fake_gen), y_fake_gen
                )
                loss_disc_fake = cross_entropy_loss(torch.zeros_like(y_fake), y_fake)
                loss_disc_real = cross_entropy_loss(torch.ones_like(y_real), y_real)
                loss_disc = loss_disc_fake + loss_disc_real + gamma * loss_disc_fake_gen

                # compute gradients
                accelerator.backward(loss_disc)

                opt_discr.step()
                # scd_discr.step()

                epoch_disc_loss += loss_disc.item()

            run.log(
                {
                    f"disc_loss/{self.symbol}": epoch_disc_loss,
                    # f"reconstruction_lr/{self.symbol}": last_lr,
                    f"disc_epoch_time/{self.symbol}": time.time() - epoch_time,
                    "epoch": epoch,
                }
            )

        print(f"Finished training {self.symbol}!")

        return accelerator.unwrap_model(model)
