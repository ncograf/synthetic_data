import conditional_flow
import numpy as np
import torch
from type_converter import TypeConverter


class TestCondFlows:
    def test_simple_signal(self):
        dtype = "float32"
        data = torch.tensor(
            np.repeat(
                [[[1, 2], [2, 1], [3, 4], [9, 4], [2, 3], [2, 2], [1, 4]]],
                repeats=4,
                axis=0,
            ),
            dtype=TypeConverter.str_to_torch(dtype),
        )
        model = conditional_flow.ConditionalFlow(
            hidden_dim=2,
            dim=data.shape[2],
            conditional_dim=20,
            n_layer=1,
            num_model_layer=2,
            drop_out=0.5,
            activation="relu",
            norm="layer",
            dtype=dtype,
        )

        _, _, _ = model(data)

        _ = model.sample(2, data[0])

        dtype = "float64"
        data = torch.tensor(
            np.repeat(
                [[[1, 2], [2, 1], [3, 4], [9, 4], [2, 3], [2, 2], [1, 4]]],
                repeats=4,
                axis=0,
            ),
            dtype=TypeConverter.str_to_torch(dtype),
        )
        model = conditional_flow.ConditionalFlow(
            hidden_dim=2,
            dim=data.shape[2],
            conditional_dim=20,
            n_layer=1,
            num_model_layer=2,
            drop_out=0,
            activation="sigmoid",
            norm="layer",
            dtype=dtype,
        )

        _, _, _ = model(data)

        _ = model.sample(2, data[0])


if __name__ == "__main__":
    TestCondFlows().test_simple_signal()
