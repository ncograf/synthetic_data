import torch
import torch.nn as nn


class ConditionalFFT(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies a fourier transformation on a D-dimensional sigmal x
        and outputs DFT(x) applied on each signal

        Args:
            x (torch.Tensor): NxTxD dimensional signal

        Returns:
            torch.Tensor: Nx(freq_filter*2)xD
        """

        if x.dim() == 1:
            x = x.unsqueeze(0)

        # fouriertransform along the second axis and then
        # make sure the lower freqencies and on the left
        fft_x = torch.fft.rfft(x, dim=1)

        # extract real and imaginary part an scale
        fft_real, fft_imag = torch.real(fft_x), torch.imag(fft_x)

        # stack in the first dimension
        fft_x = torch.cat([fft_real, fft_imag], dim=1)

        return fft_x
