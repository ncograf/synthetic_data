import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset


class SP500DataSet(Dataset):
    def __init__(self, log_returns: npt.NDArray, num_elements: int, seq_len: int):
        self.seq_len = seq_len
        self.num_elements = num_elements
        self.log_returns = log_returns

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        idx = np.random.choice(range(self.log_returns.shape[1]), size=1)
        log_returns = self.log_returns[:, idx]
        log_returns = log_returns[~np.isnan(log_returns)]
        start_idx = np.random.randint(0, log_returns.size - self.seq_len)
        end_idx = start_idx + self.seq_len

        y_real = 0
        y_fake = 1
        return torch.tensor(log_returns[start_idx:end_idx]), torch.tensor(
            [y_real, y_fake]
        )


class BatchedDataset(Dataset):
    def __init__(self, log_returns: npt.NDArray, seq_len: int):
        self.seq_len = seq_len
        self.log_returns = np.asarray(log_returns)
        self.log_returns[~np.isnan(log_returns)] = 1
        self.batch_size = 24

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        idx = np.random.choice(range(self.log_returns.shape[1]), size=self.batch_size)
        data_len = self.log_returns.shape[0]
        if data_len > self.seq_len:
            start_idx = np.random.randint(0, self.log_returns.shape[0] - self.seq_len)
        else:
            start_idx = 0
        end_idx = start_idx + self.seq_len
        log_returns = self.log_returns[start_idx:end_idx, idx]
        return torch.tensor(log_returns).T


class MixedDataSet(Dataset):
    def __init__(
        self,
        real_log_returns: npt.NDArray,
        syn_log_returns: npt.NDArray,
        num_elements: int,
        seq_len: int,
    ):
        self.seq_len = seq_len
        self.num_elements = num_elements
        self.real_log = real_log_returns
        self.syn_log = syn_log_returns

    def __len__(self):
        return self.num_elements

    def __getitem__(self, _):
        idx_real = np.random.choice(range(self.real_log.shape[1]), size=1)
        real_returns = self.real_log[:, idx_real]
        real_returns = real_returns[~np.isnan(real_returns)]
        start_real = np.random.randint(0, real_returns.size - self.seq_len)
        end_real = start_real + self.seq_len

        idx_syn = np.random.choice(range(self.syn_log.shape[1]), size=1)
        syn_returns = self.syn_log[:, idx_syn]
        syn_returns = syn_returns[~np.isnan(syn_returns)]
        start_syn = np.random.randint(0, syn_returns.size - self.seq_len)
        end_syn = start_syn + self.seq_len

        y_real = 0
        y_syn = 1

        real_data = torch.tensor(real_returns[start_real:end_real])
        syn_data = torch.tensor(syn_returns[start_syn:end_syn])

        return real_data, syn_data, torch.tensor([y_real, y_syn])
