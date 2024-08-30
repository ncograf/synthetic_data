import accelerate
import datasets
import load_data
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import train_fingan


class ConvDisc(nn.Module):
    def __init__(self, input_dim: int, dtype: torch.dtype):
        nn.Module.__init__(self)
        in_channels = 1
        input_dim = input_dim
        kernel_size = 10
        self.mod = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=32,
                padding="same",
                kernel_size=kernel_size,
                dtype=dtype,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                padding="same",
                kernel_size=kernel_size,
                dtype=dtype,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                padding="same",
                kernel_size=kernel_size,
                dtype=dtype,
            ),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(input_dim * 64, 32, dtype=dtype),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(32, 1, dtype=dtype),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)

        return self.mod(x)


class RNN(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        hidden_dim = 64
        n_layer = 2
        # self.rnn_layer = torch.nn.LSTM(1, hidden_dim, n_layer, batch_first=True, dropout=0, bidirectional=False)
        self.rnn_layer = torch.nn.GRU(1, hidden_dim, n_layer)
        self.decision_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x, _ = self.rnn_layer(x)

        # pick last element
        if x.ndim == 3:
            x = x[:, -1, :]
        else:
            x = x[-1, :]

        x = torch.sigmoid(self.decision_layer(x))

        return x


N_TICKS = 9216

epochs = 50
batch_size = 128
num_batches = 256
seq_len = 128
torch_dtype = torch.float32
numpy_dtype = np.float32

sp500 = load_data.load_log_returns("sp500").astype(numpy_dtype)
fingan_ = train_fingan.load_fingan(
    "/home/nico/thesis/code/data/cache/results/epoch_43/model.pt"
)


def fingan(S):
    return train_fingan.sample_fingan(model=fingan_, batch_size=S)


n_train = sp500.shape[1] // 2
real_train = sp500[:, :-n_train]
real_test = sp500[:, n_train:]

syn_train = fingan(n_train).astype(numpy_dtype)
syn_test = fingan(n_train).astype(numpy_dtype)

train_dataset = datasets.MixedDataSet(
    real_train, syn_train, batch_size * num_batches, seq_len
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size, shuffle=None, pin_memory=True
)

test_dataset = datasets.MixedDataSet(
    real_test, syn_test, batch_size * num_batches, seq_len
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size, shuffle=None, pin_memory=True
)

accelerate.utils.set_seed(1023)

# model = RNN().to(torch_dtype)
model = ConvDisc(seq_len, torch_dtype)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

accelerator = accelerate.Accelerator()
train_loader, model, optimizer, test_loader, scheduler = accelerator.prepare(
    train_loader, model, optimizer, test_loader, scheduler
)

# train model
for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_acc = 0.0

    for real_batch, syn_batch, y in train_loader:
        optimizer.zero_grad()

        y_real, y_syn = y[:, 0].to(torch_dtype), y[:, 1].to(torch_dtype)
        pred_syn = model(syn_batch).flatten()
        pred_real = model(real_batch).flatten()

        loss = torch.nn.functional.binary_cross_entropy(pred_real, y_real)
        loss += torch.nn.functional.binary_cross_entropy(pred_syn, y_syn)

        accelerator.backward(loss)
        optimizer.step()

        pred_real = pred_real >= 0.5
        pred_syn = pred_syn >= 0.5

        epoch_acc += torch.sum(pred_syn == y_syn).item()
        epoch_acc += torch.sum(pred_real == y_real).item()

        epoch_loss += loss.item()

    scheduler.step()

    epoch_loss = epoch_loss / len(train_loader)
    epoch_acc = epoch_acc / (2 * len(train_loader.sampler))

    print(f"Epoch {epoch + 1} / {epochs} loss: {epoch_loss}, acc : {epoch_acc}")


# evaluate tests

epoch_acc = 0
for real_batch, syn_batch, y in test_loader:
    y_real, y_syn = y[:, 0].to(torch_dtype), y[:, 1].to(torch_dtype)
    pred_real = model(real_batch).flatten()
    pred_syn = model(syn_batch).flatten()

    pred_real = pred_real >= 0.5
    pred_syn = pred_syn >= 0.5

    epoch_acc += torch.sum(pred_syn == y_syn).item()
    epoch_acc += torch.sum(pred_real == y_real).item()

epoch_acc = epoch_acc / (2 * len(test_loader.sampler))

print(f"Test accuracy is equal to {epoch_acc}")
