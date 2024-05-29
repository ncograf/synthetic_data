import torch
import torch.nn as nn


class FourierTransformLayer(nn.Module):
    def __init__(self, seq_len: int):
        nn.Module.__init__(self)

        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies a fourier transformation on a D-dimensional sigmal x
        and outputs DFT(x) applied on each signal

        Args:
            x (torch.Tensor): Bxseq_len dimensional signal

        Returns:
            torch.Tensor: Dx(T // 2 + 1)x2 freqency tensor
        """

        if x.dim() == 1:
            x = x.unsqueeze(0)

        # fouriertransform along the second axis and then
        # make sure the lower freqencies and on the left
        assert x.shape[1] == self.seq_len
        fft_x = torch.fft.rfft(x, dim=1, norm="forward")

        # extract real and imaginary part an scale
        fft_real, fft_imag = torch.real(fft_x), torch.imag(fft_x)

        # stack in the first dimension
        fft_x = torch.stack([fft_real, fft_imag], dim=-1)

        return fft_x

    def inverse(self, fft_x: torch.Tensor) -> torch.Tensor:
        """Inverse fft applied on frequency tensor

        Args:
            fft_x (torch.Tensor): Dx(T // 2 + 1)x2 freqency tensor with real and imag componentes

        Returns:
            torch.Tensor: DxT signal vector
        """

        assert fft_x.dim() == 3

        # get real an imaginary component and extend the cutted parts
        fft_real, fft_imag = fft_x[:, :, 0], fft_x[:, :, 1]

        # fouriertransform along the second axis and then
        # make sure the lower freqencies and on the left
        x = torch.fft.irfft(
            fft_real + 1j * fft_imag, n=self.seq_len, dim=1, norm="forward"
        )

        return x
