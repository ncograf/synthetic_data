import accelerate
import datasets
import load_data
import numpy as np
import torch
import torch.utils
import torch.utils.data
import train_fingan


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
        x = torch.sigmoid(self.decision_layer(x))

        return x


N_TICKS = 9216

epochs = 50
batch_size = 256
num_batches = 64
seq_len = 256
torch_dtype = torch.float32
numpy_dtype = np.float32

sp500 = load_data.load_log_returns("sp500").astype(numpy_dtype)
fingan_ = train_fingan.load_fingan(
    "/home/nico/thesis/code/data/cache/results/epoch_43/model.pt"
)


def fingan(S):
    return train_fingan.sample_fingan(model=fingan_, batch_size=S)


fake_train = fingan(256).astype(numpy_dtype)

real_train = sp500[: -(seq_len + 1), :]
real_test = sp500[-(seq_len + 1) :, :]

length = np.minimum(fake_train.shape[0], real_train.shape[0])
mixed_train = np.concatenate([fake_train[-length:], real_train[-length:]], axis=1)

real_train_dataset = datasets.SP500DataSet(
    real_train, batch_size * num_batches, seq_len + 1
)
real_train_loader = torch.utils.data.DataLoader(
    real_train_dataset, batch_size, shuffle=None, pin_memory=True
)

fake_train_dataset = datasets.SP500DataSet(
    fake_train, batch_size * num_batches, seq_len + 1
)
fake_train_loader = torch.utils.data.DataLoader(
    fake_train_dataset, batch_size, shuffle=None, pin_memory=True
)

mixed_train_dataset = datasets.SP500DataSet(
    mixed_train, batch_size * num_batches, seq_len + 1
)
mixed_train_loader = torch.utils.data.DataLoader(
    mixed_train_dataset, batch_size, shuffle=None, pin_memory=True
)

real_test_dataset = torch.utils.data.TensorDataset(torch.tensor(real_test).T)
test_loader = torch.utils.data.DataLoader(real_test_dataset, batch_size, shuffle=False)


accelerate.utils.set_seed(1023)

model = RNN().to(torch_dtype)
train_loader = real_train_loader
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

accelerator = accelerate.Accelerator()
train_loader, model, optimizer, test_loader, scheduler = accelerator.prepare(
    train_loader, model, optimizer, test_loader, scheduler
)

# train model
for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_mape = 0.0

    for batch, _ in train_loader:
        optimizer.zero_grad()

        batch = batch**2
        batch = batch / torch.std(batch)

        labels = batch[:, -seq_len // 2 :]
        batch = batch[:, :-1]

        pred = model(batch)[:, -seq_len // 2 :, 0]

        loss = torch.sum((labels - pred) ** 2)

        mape = torch.sum(torch.abs((labels - pred) / (labels + 1e-10)))

        accelerator.backward(loss)

        optimizer.step()

        epoch_loss += loss.item()
        epoch_mape += mape.item()

    scheduler.step()

    epoch_loss = epoch_loss / (len(train_loader.sampler) * seq_len / 2)
    epoch_mape = epoch_mape / (len(train_loader.sampler) * seq_len / 2)

    print(f"Epoch {epoch + 1} / {epochs} loss: {epoch_loss}, mape : {epoch_mape}")


# evaluate tests

mape = 0
for (batch,) in test_loader:
    batch = batch**2
    # batch = batch / (torch.max(batch) - torch.min(batch))

    labels = batch[:, -seq_len // 2 :]
    batch = batch[:, :-1]
    pred = model(batch)[:, -seq_len // 2 :, 0]

    mape = torch.sum(torch.abs((labels - pred) / (labels + 1e-10)))

mape = mape / (len(train_loader.sampler) * seq_len / 2)

print(f"MAPE is equal to {mape}")
