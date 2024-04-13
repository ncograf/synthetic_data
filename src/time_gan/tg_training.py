from typing import Dict, List, Tuple

import numpy as np
import torch
from tg_rnn_network import TGRNNNetwork
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset


def sample_noise(B: int, T: int, dim: int) -> torch.Tensor:
    """Compute noise for the generator input

    Args:
        B (int): Batch size
        T (int): Time series length
        dim (int): dimension of latent variable

    Returns:
        torch.Tensor: Sampled noise tensor
    """
    noise = torch.rand(size=(B, T, dim))
    return noise


def min_max_scale(x: torch.Tensor):
    min = torch.min(x)
    x_ = x - min

    amplitude = torch.max(x_)

    x_ = x_ / (amplitude + 1e-10)

    return x_, (min, amplitude)


def min_max_inverse(x: torch.Tensor, min_and_amplitude: Tuple[float, float]):
    min, amplitude = min_and_amplitude

    x_ = x * amplitude
    x_ = x + min

    return x_


def train_time_gan(
    X: torch.Tensor, epochs: int, learning_rate: float
) -> Tuple[TGRNNNetwork, Dict[str, List[float]]]:
    D = X.shape[0]
    T = X.shape[1]
    z_dim = 1
    hidden_dim = 24
    n_layer = 3
    batch_size = 128

    gamma = 1

    X_, min_ampl = min_max_scale(X)

    data_set = TensorDataset(X_)
    data_loader = DataLoader(data_set, batch_size=batch_size)

    # Define necessary networks
    embedder = TGRNNNetwork(
        T, hidden_dim=hidden_dim, num_layer=n_layer, in_dim=1, out_dim=1, bidirect=False
    )
    recovery = TGRNNNetwork(
        T, hidden_dim=hidden_dim, num_layer=n_layer, in_dim=1, out_dim=1, bidirect=False
    )
    # The supervisor is a hack to make the teacher forcing work
    supervisor = TGRNNNetwork(
        T, hidden_dim=hidden_dim, num_layer=n_layer, in_dim=1, out_dim=1, bidirect=False
    )
    generator = TGRNNNetwork(
        T,
        hidden_dim=hidden_dim,
        num_layer=n_layer,
        in_dim=z_dim,
        out_dim=1,
        bidirect=False,
    )
    discriminator = TGRNNNetwork(
        T, hidden_dim=hidden_dim, num_layer=n_layer, in_dim=1, out_dim=1, bidirect=True
    )

    cross_entropy_loss = CrossEntropyLoss()

    # Define optimizer
    recon_params = list(embedder.parameters()) + list(recovery.parameters())
    opt_recon = torch.optim.Adam(recon_params, lr=learning_rate)

    super_params = list(supervisor.parameters())
    opt_super = torch.optim.Adam(super_params, lr=learning_rate)

    gen_params = list(generator.parameters()) + recon_params + super_params
    opt_gener = torch.optim.Adam(gen_params, lr=learning_rate)

    discr_params = list(discriminator.parameters())
    opt_discr = torch.optim.Adam(discr_params, lr=learning_rate)

    # Define scheduler functions
    # scd_recon = torch.optim.lr_scheduler.ExponentialLR(opt_recon, gamma=0.999)
    # scd_super = torch.optim.lr_scheduler.ExponentialLR(opt_super, gamma=0.999)
    # scd_gener = torch.optim.lr_scheduler.ExponentialLR(opt_gener, gamma=0.999)
    # scd_discr = torch.optim.lr_scheduler.ExponentialLR(opt_discr, gamma=0.999)

    losses = {
        "recon": [],
        "super": [],
        "gener": [],
        "discr": [],
        "embed": [],
    }

    # train reconstruction network part
    for epoch in range(epochs):
        epoch_loss = 0

        for (x,) in data_loader:
            opt_recon.zero_grad()

            h = embedder(x)
            x_tilde = recovery(h)
            loss = torch.mean(
                (x_tilde.flatten() - x.flatten()) ** 2
            )  # reconstructed x and real x
            loss.backward()
            opt_recon.step()
            # scd_recon.step()

            epoch_loss += loss.item()

        losses["recon"].append(epoch_loss)

        if epoch % 10 == 0:
            print(
                (
                    f'Reconstruction Epoch: {epoch:>10d}, last loss {loss.item():>15.4f},'
                    f' average_loss {np.mean(losses["recon"]):>10.4f}'
                )
            )

    # train with supervised lossx.shape + (1)
    # note that this is a hack needed to replace teacher forcing
    for epoch in range(epochs):
        epoch_loss = 0

        for (x,) in data_loader:
            opt_super.zero_grad()

            h = embedder(x)
            h_super = supervisor(h)
            loss = torch.mean((h[:, :-1] - h_super[:, 1:]) ** 2)  # supervisor loss
            loss.backward()
            opt_super.step()
            # scd_super.step()

            epoch_loss += loss.item()

        losses["super"].append(epoch_loss)

        if epoch % 10 == 0:
            print(
                (
                    f'Supervised Epoch: {epoch:>10d}, last loss {loss.item():>15.4f},'
                    f' average_loss {np.mean(losses["super"]):>10.4f}'
                )
            )

    # GAN training
    for epoch in range(epochs):
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        epoch_super_loss = 0
        n_gen = 3

        # Train the generator twice
        for _ in range(n_gen):
            for (x,) in data_loader:
                opt_gener.zero_grad()

                z = sample_noise(D, T, z_dim)
                h_real = embedder(x)
                h_gen = generator(z)
                h_hat = supervisor(
                    h_gen
                )  # generator and supervisor are used to enable teacher forcing
                h_hat_super = supervisor(h_real)

                x_hat = recovery(h_hat)  # get back the x from the generator
                x_hat_real = recovery(h_real)  # get back the x from the supervisor

                y_fake = discriminator(h_hat)
                y_fake_gen = discriminator(h_gen)

                # embedder and supervised loss
                loss_supervisor = torch.mean(
                    (h_real[:, :-1] - h_hat_super[:, 1:]) ** 2
                )  # supervisor loss
                loss_embedder = torch.mean(
                    (x.flatten() - x_hat_real.flatten()) ** 2
                )  # embedder loss
                loss_emb_sup = 10 * torch.sqrt(loss_embedder) + 0.1 * loss_supervisor

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
                loss_all.backward()

                opt_gener.step()
                # scd_gener.step()

                epoch_gen_loss += loss_generator.item()
                epoch_super_loss += loss_emb_sup.item()

        epoch_gen_loss /= n_gen
        epoch_super_loss /= n_gen

        for (x,) in data_loader:
            opt_discr.zero_grad()

            z = sample_noise(D, T, z_dim)
            h_real = embedder(x)
            h_gen = generator(z)
            h_hat = supervisor(
                h_gen
            )  # generator and supervisor are used to enable teacher forcing

            y_fake = discriminator(h_hat)
            y_fake_gen = discriminator(h_gen)
            y_real = discriminator(
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
            loss_disc.backward()

            opt_discr.step()
            # scd_discr.step()

            epoch_disc_loss += loss_disc.item()

        losses["discr"].append(epoch_disc_loss)
        losses["gener"].append(epoch_gen_loss)
        losses["embed"].append(epoch_super_loss)

        if epoch % 10 == 0:
            print(f"Joint training epoch: {epoch:>10d}")
            print(
                (
                    f'{"Last Embedding loss":<25} {loss_emb_sup.item():>10.4f},'
                    f' aveage_loss {np.mean(losses["embed"]):>10.4f}'
                )
            )
            print(
                (
                    f'{"Last Discriminator loss":<25} {loss_disc.item():>10.4f},'
                    f' aveage_loss {np.mean(losses["discr"]):>10.4f}'
                )
            )
            print(
                (
                    f'{"Last Generator loss":<25} {loss_generator.item():>10.4f},'
                    f' aveage_loss {np.mean(losses["gener"]):>10.4f}'
                )
            )
            print("")  # empty line

    return (embedder, recovery, generator, supervisor, discriminator), losses


if __name__ == "__main__":
    print("Start test")
    n_samples = 20
    T = 24

    freqs = np.random.beta(a=2, b=2, size=n_samples).reshape((-1, 1))
    phases = np.random.normal(size=n_samples).reshape((-1, 1))

    signals = np.repeat(
        np.reshape(np.arange(T, dtype=np.float32), (1, -1)), repeats=n_samples, axis=0
    )
    signals = np.sin(signals * freqs + phases)

    X_signal = torch.tensor(signals, dtype=torch.float32)

    models, losses = train_time_gan(X_signal, 101, 1e-3)
    print("End test")
