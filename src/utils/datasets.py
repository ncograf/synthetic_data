import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SP500Dataset(Dataset):
    def __init__(self, price_data: pd.DataFrame, num_elements: int, seq_len: int):
        self.seq_len = seq_len
        self.num_elements = num_elements

        # all colums in the dataframe must have at least seq_len non_nan elements
        non_nans = np.array(np.sum(~np.isnan(price_data), axis=0))
        self.price_data = price_data.drop(non_nans <= seq_len)

    def __len__(self):
        return self.num_elements

    def min_max_scaling(self, data: torch.Tensor) -> torch.Tensor:
        """Min max scaling of data

        Args:
            data (torch.Tensor): Data to be scaled

        Returns:
            Tuple[torch.Tensor, float, float]: scaled data, shift, scale
        """

        shift = torch.min(data)
        data_ = data - shift
        scale = torch.max(data_)

        data_ = data_ / scale

        return data_, shift, scale

    def __getitem__(self, idx):
        # choose random symbol until one has enough data
        symbol = np.random.choice(self.price_data.columns, size=1)
        data = np.array(self.price_data.loc[:, symbol].dropna())
        log_returns = np.log(data[1:] / data[:-1])

        start_idx = np.random.randint(0, log_returns.size - self.seq_len)
        end_idx = start_idx + self.seq_len

        return torch.tensor(log_returns[start_idx:end_idx])


class SP500GanDataset(Dataset):
    def __init__(self, price_data: pd.DataFrame, num_elements: int, seq_len: int):
        self.seq_len = seq_len
        self.num_elements = num_elements

        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame()

        # all colums in the dataframe must have at least seq_len non_nan elements
        non_nans = np.array(np.sum(~np.isnan(price_data), axis=0))
        self.price_data = price_data.drop(
            price_data.columns[non_nans <= seq_len], axis="columns"
        )

        # choose random symbol until one has enough data
        data = np.array(self.price_data)
        self.log_returns = np.log(data[1:] / data[:-1])

        self.shift = 0  # np.nanmean(self.log_returns)
        self.scale = 1  # np.nanstd(self.log_returns)
        self.log_returns = self.log_returns - self.shift
        self.log_returns = self.log_returns / self.scale

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        idx = np.random.choice(range(self.log_returns.shape[1]), size=1)
        log_returns = self.log_returns[:, idx]
        log_returns = log_returns[~np.isnan(log_returns)]
        start_idx = np.random.randint(0, log_returns.size - self.seq_len)
        end_idx = start_idx + self.seq_len

        y_real = torch.rand(size=(1,)) * 0.2
        y_fake = torch.rand(size=(1,)) * 0.2 + 0.8
        return torch.tensor(log_returns[start_idx:end_idx]), torch.tensor(
            [y_real, y_fake]
        )
