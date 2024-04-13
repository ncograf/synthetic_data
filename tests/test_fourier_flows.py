import torch
import numpy as np
import fourier_flow


class TestFourierFlows:
    def test_simple_signal(self):
        data = torch.tensor(np.repeat([[1, 2, 3, 1, 2, 3, 1]], repeats=4, axis=0))
        model = fourier_flow.FourierFlow(1, D=data.shape[0], T=data.shape[1], n_layer=1)

        _, _, _ = model(data)

        _ = model.sample(2)


if __name__ == "__main__":
    TestFourierFlows().test_simple_signal()
