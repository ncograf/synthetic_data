import torch
import torch.nn as nn

class FourierTransformLayer(nn.Module):
    
    def __init__(self, T : int):
        
        nn.Module.__init__(self)
        
        self.T = T
        
        if T % 2 == 0:
            raise ValueError("Fourier Transforms of even seqence lenghts are not supported")
        
        # if self.n_freq % 2 == 0 the output will be larger by TWO (than input)
        # if self.n_freq % 2 == 1 the output will be larger by ONE (than input)
        # Add one as the constant frequency must be present
        self.n_non_redundant = self.T // 2 + 1
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """Applies a fourier transformation on a D-dimensional sigmal x
        and outputs DFT(x) applied on each signal

        Args:
            x (torch.Tensor): DxT dimensional signal

        Returns:
            torch.Tensor: Dx(T // 2 + 1)x2 freqency tensor
        """
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # fouriertransform along the second axis and then
        # make sure the lower freqencies and on the left
        fft_x = torch.fft.fftshift(torch.fft.fft(x,dim=1), dim=1)
        
        # crop the redundant data
        fft_x = fft_x[:, :self.n_non_redundant]
        
        # extract real and imaginary part an scale
        fft_real, fft_imag = torch.real(fft_x) / self.T, torch.imag(fft_x) / self.T
        
        # stack in the first dimension
        fft_x = torch.stack([fft_real, fft_imag], dim=-1)
        
        return fft_x

    def inverse(self, fft_x : torch.Tensor) -> torch.Tensor:
        """Inverse fft applied on frequency tensor

        Args:
            fft_x (torch.Tensor): Dx(T // 2 + 1)x2 freqency tensor with real and imag componentes

        Returns:
            torch.Tensor: DxT signal vector
        """
        
        assert fft_x.dim() == 3
        
        # get real an imaginary component and extend the cutted parts
        fft_real, fft_imag = fft_x[:,:,0], fft_x[:,:,1]
        fft_real = torch.cat([fft_real, torch.flip(fft_real, dims=[1])[:,1:]], dim=1)
        fft_imag = torch.cat([fft_imag, -torch.flip(fft_imag, dims=[1])[:,1:]], dim=1)
        
        # fouriertransform along the second axis and then
        # make sure the lower freqencies and on the left
        x = torch.real(torch.fft.ifft(torch.fft.ifftshift(fft_real + 1j * fft_imag, dim=1), dim=1)) * self.T

        return x