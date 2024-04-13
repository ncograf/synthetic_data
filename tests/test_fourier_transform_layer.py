import fourier_transform_layer
import numpy as np
import torch


class TestFourierTransformLayer:
    def test_simple_signal_odd(self):
        layer = fourier_transform_layer.FourierTransformLayer(7)

        data = torch.tensor([[1, 2, 3, 1, 2, 3, 1]])

        trans = layer.forward(data)

        reconst = layer.inverse(trans)

        assert np.all(np.abs(np.array(reconst - data)) < 1e-6)

    def test_random(self):
        T = 11
        D = 20
        layer = fourier_transform_layer.FourierTransformLayer(T)

        data = torch.rand((D, T))

        trans = layer.forward(data)

        reconst = layer.inverse(trans)

        assert np.all(np.abs(np.array(reconst - data)) < 1e-6)


if __name__ == "__main__":
    TestFourierTransformLayer().test_simple_signal_odd()
    TestFourierTransformLayer().test_random()
