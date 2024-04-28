import fourier_flow
import numpy as np
import torch
from type_converter import TypeConverter


class TestFourierFlows:
    def test_simple_signal(self):
        dtype = "float32"
        data = torch.tensor(
            np.repeat([[1, 2, 3, 1, 2, 3, 1]], repeats=4, axis=0),
            dtype=TypeConverter.str_to_torch(dtype),
        )
        model = fourier_flow.FourierFlow(1, T=data.shape[1], n_layer=1, dtype=dtype)

        _, _, _ = model(data)

        _ = model.sample(2)

        dtype = "float64"
        data = torch.tensor(
            np.repeat([[1, 2, 3, 1, 2, 3, 1]], repeats=4, axis=0),
            dtype=TypeConverter.str_to_torch(dtype),
        )
        model = fourier_flow.FourierFlow(1, T=data.shape[1], n_layer=1, dtype=dtype)

        _, _, _ = model(data)

        _ = model.sample(2)


if __name__ == "__main__":
    TestFourierFlows().test_simple_signal()
