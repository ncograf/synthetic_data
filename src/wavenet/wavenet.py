from typing import Any, Dict, List

import datasets
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.utils.data.sampler


class CausalConv(nn.Module):
    def __init__(self, channels: List[int]):
        nn.Module.__init__(self)

        kernel_size = 2

        in_channels = [1] + channels[:-1]
        out_channels = channels
        self.layers = nn.ModuleList()
        for in_, out_ in zip(in_channels, out_channels):
            conv = nn.Conv1d(
                in_,
                out_,
                kernel_size=kernel_size,
                stride=1,  # DO NOT CHANGE (Skip + res connections)
                padding=0,
            )
            self.layers.append(conv)

        self.padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-2)

        for layer in self.layers:
            x = layer(x)
            x = nn.functional.pad(x, self.padding)

        return x


class DelatedCausalConv(nn.Module):
    def __init__(self, in_channels: int, res_channels: int, layer: int):
        nn.Module.__init__(self)

        self.kernel_size = 2

        in_conv = nn.Conv1d(
            in_channels,
            res_channels,
            kernel_size=self.kernel_size,
            stride=1,  # DO NOT CHANGE (Skip + res connections)
            padding=0,
            dilation=1,
        )
        self.layers = nn.ModuleList([in_conv])
        for i in range(1, layer):
            conv = nn.Conv1d(
                res_channels,
                res_channels,
                kernel_size=self.kernel_size,
                stride=1,  # DO NOT CHANGE (Skip + res connections)
                padding=0,
                dilation=2**i,
            )
            self.layers.append(conv)

        self.layers.append(
            nn.Conv1d(
                res_channels,
                in_channels,
                kernel_size=self.kernel_size,
                stride=1,  # DO NOT CHANGE (Skip + res connections)
                padding=0,
                dilation=2**layer,
            )
        )

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers, start=0):
            x = nn.functional.pad(x, ((self.kernel_size - 1) * (2**i), 0))
            x = layer(x)

        return x


class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels: int, res_channels: int, layer: int):
        nn.Module.__init__(self)
        self.g = DelatedCausalConv(in_channels, res_channels, layer)
        self.f = DelatedCausalConv(in_channels, res_channels, layer)
        self.scale = nn.Conv1d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor):
        z = torch.tanh(self.f(x)) * torch.sigmoid(self.g(x))

        return self.scale(z) + x  # add residual


class StackedGates(nn.Module):
    def __init__(
        self, channels: List[int], res_channels: int, dil_layer: int, num_blocks: int
    ):
        nn.Module.__init__(self)

        self.gates = [
            GatedResidualBlock(channels[-1], res_channels, dil_layer)
            for _ in range(num_blocks)
        ]
        self.conv = CausalConv(channels)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)

        skip = []
        for gate in self.gates:
            x = gate(x)
            skip.append(x)

        x = torch.sum(torch.stack(skip, dim=-1), dim=-1)

        return x


class WaveNet(nn.Module):
    def __init__(
        self, stacks: List[Dict[str, Any]], classes: int, sample_data: npt.ArrayLike
    ):
        nn.Module.__init__(self)

        self.dataset = datasets.BatchedDataset(
            sample_data, np.min([512, sample_data.shape[0]])
        )

        self._stacks = stacks
        self.classes = classes
        self.scale = 1
        self.sample_data = sample_data

        self.stacks = nn.ModuleList()
        for stack in stacks:
            stack["channels"].append(classes)
            self.stacks.append(StackedGates(**stack))

        self.conv_one = nn.Conv1d(classes, classes, 1)
        self.conv_two = nn.Conv1d(classes, classes, 1)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = x / self.scale
        x = x.clamp(-1, 1)
        x = (
            torch.sign(x)
            * torch.log(1 + self.classes * torch.abs(x))
            / np.log(1.0 + self.classes)
        )

        return x

    def inv_transform(self, x: torch.Tensor) -> torch.Tensor:
        x = (
            torch.sign(x)
            * (torch.exp(np.log(1.0 + self.classes) * torch.abs(x)) - 1.0)
            / self.classes
        )
        x = x.clamp(-1, 1)
        x = x * self.scale

        return x

    def forward(self, x: torch.Tensor):
        x = self.transform(x)

        x_ = []
        for stack in self.stacks:
            x_.append(stack(x))

        x = torch.sum(torch.stack(x_, dim=-1), dim=-1)
        x = torch.relu(x)
        x = torch.relu(self.conv_one(x))
        x = torch.softmax(self.conv_two(x), dim=-2)

        return x

    def get_model_info(self) -> Dict[str, Any]:
        """Model initialization parameters

        Returns:
            Dict[str, Any]: dictonary for model initialization
        """

        dict_ = {
            "stacks": self._stacks,
            "classes": self.classes,
            # "scale": self.scale,
            "sample_data": self.sample_data,
        }

        return dict_

    def sample(self, n: int, seq_len: int) -> torch.Tensor:
        """Sample new series from the learn distribution with
        given initial series x

        Args:
            n (int) : number of batches to sample
            seq_len (int): series length to sample

        Returns:
            Tensor: signals in the signal space
        """

        self.dataset.set_batch_size(n)
        data = torch.as_tensor(self.dataset[0], dtype=torch.float32)

        space = torch.linspace(self.scale[0], self.scale[1], self.classes)
        space = self.inv_transform(space)

        seq = []
        for _ in range(seq_len):
            class_probs = self.forward(data)[:, :, -1]
            idx = torch.multinomial(class_probs, 1).flatten()
            samples = space[idx].reshape((-1, 1))
            data[:, :-1] = data[:, 1:]
            data[:, -1:] = samples
            seq.append(samples)

        return torch.stack(seq, dim=-1).squeeze()


if __name__ == "__main__":
    classes = 4
    t = torch.ones([300, 200])
    test = WaveNet(
        [{"channels": [3, 4], "res_channels": 2, "dil_layer": 5, "num_blocks": 2}],
        classes,
        t,
    )
    o = test(t)
    assert o.shape[0] == 300
    assert o.shape[1] == classes
    assert o.shape[2] == 200
    assert o.ndim == 3

    o = test.sample(300, 200)
    assert o.shape[0] == t.shape[0]
    assert o.shape[1] == 200

    t = torch.ones([200])
    o = test(t)
    assert o.shape[0] == classes

    t = torch.rand((10))
    t = t * 2 - 1
    t_ = test.inv_transform(test.transform(t))
    assert torch.all(torch.abs(t - t_) < 1e-4)
