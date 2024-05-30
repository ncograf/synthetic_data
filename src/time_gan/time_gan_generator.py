import time
from typing import Any, Dict, Tuple

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


class TimeGanGenerator(base_generator.BaseGenerator):
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
                time_gan_config:
                    hidden_dim: int
                    num_layer: int
                    embed_dim: int
                    n_stock: int
                seq_len: int
                dtype: str
                epochs: int
                batch_size: int
                lag: int
                gamma (float): Factor for exponential LRScheduler (in every epoch the new learning rate is lr * gamma)
                optim_config: config for adam optimizer (e.g. lr : float)
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
        lag = config["lag"]
        dtype = TypeConverter.str_to_torch(config["dtype"])

        # transform dataset for training
        data = torch.from_numpy(price_data.values)
        data_mask = ~torch.isnan(data)
        data = data[data_mask]
        _zero_price = data[0]
        log_returns = torch.log(data[1:] / data[:-1])

        # split the data
        X = np.lib.stride_tricks.sliding_window_view(
            log_returns, seq_len, axis=0, writeable=False
        )
        X = torch.tensor(X[::lag], dtype=dtype)
        X = X.unsqueeze(dim=-1)

        X, shift, scale = self.min_max_scaling(X)

        # train model
        _model, score = self.train_time_gan(
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

        model = TimeGan(**config["init_params"])
        model.load_state_dict(config["state_dict"])

        model_output = model.sample(length, burn=burn)
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

    def train_time_gan(
        self,
        X: torch.Tensor,
        config: Dict[str, Any],
        accelerator: Accelerator,
        sym: str = "",
    ) -> TimeGan:
        """Train a Time Gan network with given parameters

        Args:
            run (wr.Run): Wandb run for logging.
            X (torch.Tensor): Training data
            config (Dict[str, Any]): configuration for training run:
                time_gan_config:
                    hidden_dim: int
                    num_layer: int
                    embed_dim: int
                    n_stock: int
                seq_len: int
                dtype: str
                epochs: int
                batch_size: int
                lag: int
                gamma (float): Factor for exponential LRScheduler (in every epoch the new learning rate is lr * gamma)
                optim_config: config for adam optimizer (e.g. lr : float)
                lr_config : config for exponential lr_scheduler (e.g. gamma : float)
            accelerator (Accelerator): For fast training
            sym (str, optional) : symbol for logging. Defaults to ""

        Returns:
            Tuple[TimeGan, float]: Trained network ready for sampling, final sum of generator and discriminator loss
        """

        model_tg = TimeGan(**config["time_gan_config"], dtype=config["dtype"])

        batch_size = config["batch_size"]
        gamma = config["gamma"]
        epochs = config["epochs"]
        # log gradient information
        # run.watch(model, log="all")

        dataset = TensorDataset(X)
        loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

        # Define optimizer
        recon_params = list(model_tg.embedder.parameters()) + list(
            model_tg.recovery.parameters()
        )
        opt_recon = torch.optim.Adam(recon_params, **config["optim_config"])

        super_params = list(model_tg.supervisor.parameters())
        opt_super = torch.optim.Adam(super_params, **config["optim_config"])

        gen_params = list(model_tg.generator.parameters()) + recon_params + super_params
        opt_gener = torch.optim.Adam(gen_params, **config["optim_config"])

        discr_params = list(model_tg.discriminator.parameters())
        opt_discr = torch.optim.Adam(discr_params, **config["optim_config"])

        accelerator.free_memory()
        loader, model, opt_recon, opt_super, opt_gener, opt_discr = accelerator.prepare(
            loader, model_tg, opt_recon, opt_super, opt_gener, opt_discr
        )

        cross_entropy_loss = CrossEntropyLoss()

        # Bad configuration might make the model collaps
        assert model is not None

        # train reconstruction network part
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_time = time.time()

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

            if wandb.run is not None:
                wandb.log(
                    {
                        f"recon_loss/{sym}": epoch_loss,
                        # f"reconstruction_lr/{self.symbol}": last_lr,
                        f"recon_epoch_time/{sym}": time.time() - epoch_time,
                        "epoch": epoch,
                    }
                )

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

            if wandb.run is not None:
                wandb.log(
                    {
                        f"super_loss/{sym}": epoch_loss,
                        # f"reconstruction_lr/{self.symbol}": last_lr,
                        f"super_epoch_time/{sym}": time.time() - epoch_time,
                        "epoch": epoch,
                    }
                )

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

            if wandb.run is not None:
                wandb.log(
                    {
                        f"generator_loss/{sym}": epoch_gen_loss,
                        f"embed_loss/{sym}": epoch_gen_loss,
                        # f"reconstruction_lr/{self.symbol}": last_lr,
                        f"generator_epoch_time/{sym}": time.time() - epoch_time,
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

            if wandb.run is not None:
                wandb.log(
                    {
                        f"disc_loss/{sym}": epoch_disc_loss,
                        # f"reconstruction_lr/{self.symbol}": last_lr,
                        f"disc_epoch_time/{sym}": time.time() - epoch_time,
                        "epoch": epoch,
                    }
                )

        final_loss = epoch_disc_loss + epoch_gen_loss
        print(f"Finished training {sym}!")

        accelerator.free_memory()
        return accelerator.unwrap_model(model), final_loss
