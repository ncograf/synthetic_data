from math import ceil, exp2, log2

import torch


def _auto_corr(x: torch.Tensor, max_lag: int, dim: int = -1):
    """Compute lagged corr
    math: ```
        \sum_t x_t * x_{t+k}
    ```

    Args:
        x (torch.Tensor): input data
        max_lag (int): maximal lag to be returned (Note that this does not change the computains effort)
        dim (int, optional): Dimension along which to compute the correlation. Defaults to -1.

    Returns:
        torch.Tensor: \sum_t x_t * x_{t+k}, for 0 \leq k \leq max_lag
    """

    n_data = x.shape[dim]
    n_fft = int(
        exp2(ceil(log2(n_data)) + 1)
    )  # cuda only supports powers of 2 (+ 1 to avoid circulant computation)

    # convenience, to make fourier transform in last dimension
    x = x.transpose(dim, -1)

    fft_data = torch.fft.rfft(x, n=n_fft)
    fft_conv = torch.conj(fft_data) * fft_data  # note for real data conj(c_k) = c_{n-k}
    conv = torch.real(torch.fft.irfft(fft_conv, n=n_fft))
    conv = conv[..., : max_lag + 1]

    return conv.transpose(dim, -1)


def _lagged_corr(
    x: torch.Tensor, y: torch.Tensor, max_lag: int, dim=-1
) -> torch.Tensor:
    """Compute lagged corr
    math: ```
        \sum_t x_t * y_{t+k}
    ```

    Args:
        x (torch.Tensor): first input
        y (torch.Tensor): lagged input
        max_lag (int): maximal lag to be returned (Note that this does not change the computains effort)
        dim (int, optional): Dimension along which to compute the correlation. Defaults to -1.

    Returns:
        torch.Tensor: \sum_t x_t * y_{t+k}, for 0 \leq k \leq max_lag
    """

    n_data = max(x.shape[dim], y.shape[dim])
    n_fft = int(
        exp2(ceil(log2(n_data)) + 1)
    )  # cuda only supports powers of 2 (+ 1 to avoid circulant computation)

    # convenience, to make fourier transform in last dimension
    x = x.transpose(dim, -1)
    y = y.transpose(dim, -1)

    fft_x = torch.fft.rfft(x, n=n_fft)
    fft_y = torch.fft.rfft(y, n=n_fft)
    fft_conv = torch.conj(fft_x) * fft_y  # note for real data conj(c_k) = c_{n-k}
    conv = torch.real(torch.fft.irfft(fft_conv, n=n_fft))
    conv = conv[..., : max_lag + 1]

    return conv.transpose(dim, -1)


def auto_corr(x: torch.Tensor, max_lag: int, dim=-1) -> torch.Tensor:
    """Compute lagged auto corr
    math: ```
        1 / T \sum_t x_t * x_{t+k}
    ```

    Args:
        x (torch.Tensor): input data
        max_lag (int): maximal lag to be returned (Note that this does not change the computains effort)
        dim (int, optional): Dimension along which to compute the correlation. Defaults to -1.

    Returns:
        torch.Tensor: 1 / T \sum_t (x_t - \mu_x) * (x_{t+k} - \mu_y), for 0 \leq k \leq max_lag
    """

    # compute number of values
    num = torch.sum(~torch.isnan(x), dim=dim).unsqueeze(0).expand((max_lag + 1, -1))
    lag_penality = torch.arange(max_lag + 1).unsqueeze(1).expand(num.shape)
    num = num - lag_penality

    x = x.nan_to_num(0, 0, 0)
    mu_x = torch.mean(x, dim=dim)

    # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately
    x = x - mu_x
    corr = _auto_corr(x, max_lag=max_lag, dim=dim)
    corr /= num

    return corr


def lagged_corr(x: torch.Tensor, y: torch.Tensor, max_lag: int, dim=-1) -> torch.Tensor:
    """Compute lagged corr
    math: ```
        1 / T \sum_t x_t * y_{t+k}
    ```

    Args:
        x (torch.Tensor): first input
        y (torch.Tensor): lagged input
        max_lag (int): maximal lag to be returned (Note that this does not change the computains effort)
        dim (int, optional): Dimension along which to compute the correlation. Defaults to -1.

    Returns:
        torch.Tensor: 1 / T \sum_t (x_t - \mu_x) * (y_{t+k} - \mu_y), for 0 \leq k \leq max_lag
    """

    # compute number of values
    num_x = torch.sum(~torch.isnan(x), dim=dim).unsqueeze(0).expand((max_lag + 1, -1))
    num_y = torch.sum(~torch.isnan(y), dim=dim).unsqueeze(0).expand((max_lag + 1, -1))
    num = torch.max(num_x, num_y)
    lag_penality = torch.arange(max_lag + 1).unsqueeze(1).expand(num.shape)
    num = num - lag_penality

    x = x.nan_to_num(0, 0, 0)
    mu_x = torch.mean(x, dim=dim)

    y = y.nan_to_num(0, 0, 0)
    mu_y = torch.mean(y, dim=dim)

    # compute the (r_{t+k} - mu) part of the correlation and the (r_t - mu) part separately
    x = x - mu_x
    y = y - mu_y
    corr = _lagged_corr(x, y, max_lag=max_lag, dim=dim)
    corr /= num

    return corr
