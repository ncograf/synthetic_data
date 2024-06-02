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
        model = fourier_flow.FourierFlow(
            1,
            seq_len=data.shape[1],
            num_layer=1,
            dtype=dtype,
            num_model_layer=1,
            drop_out=0.5,
            activation="relu",
            norm="layer",
            arch="MLP",
        )

        _, _, _ = model(data)

        _ = model.sample(2)

        dtype = "float64"
        data = torch.tensor(
            np.repeat([[1, 2, 3, 1, 2, 3, 1]], repeats=4, axis=0),
            dtype=TypeConverter.str_to_torch(dtype),
        )
        model = fourier_flow.FourierFlow(
            1,
            seq_len=data.shape[1],
            num_layer=1,
            dtype=dtype,
            num_model_layer=2,
            drop_out=0,
            activation="celu",
            norm="batch",
            arch="MLP",
        )

        _, _, _ = model(data)

        _ = model.sample(2)

    def test_rnn_signal(self):
        dtype = "float32"
        data = torch.tensor(
            np.repeat([[1, 2, 3, 1, 2, 3, 1]], repeats=4, axis=0),
            dtype=TypeConverter.str_to_torch(dtype),
        )
        model = fourier_flow.FourierFlow(
            1,
            seq_len=data.shape[1],
            num_layer=1,
            dtype=dtype,
            num_model_layer=3,
            drop_out=0.9,
            activation="sigmoid",
            norm="none",
            arch="LSTM",
        )

        _, _, _ = model(data)

        _ = model.sample(2)

        dtype = "float64"
        data = torch.tensor(
            np.repeat([[1, 2, 3, 1, 2, 3, 1]], repeats=4, axis=0),
            dtype=TypeConverter.str_to_torch(dtype),
        )
        model = fourier_flow.FourierFlow(
            1,
            seq_len=data.shape[1],
            num_layer=2,
            dtype=dtype,
            num_model_layer=2,
            drop_out=0.2,
            activation="tanh",
            norm="layer",
            arch="LSTM",
        )

        _, _, _ = model(data)

        _ = model.sample(2)


if __name__ == "__main__":
    TestFourierFlows().test_rnn_signal()
