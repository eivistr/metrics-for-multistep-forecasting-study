import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm
import yaml

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

with open(os.path.join(__location__, 'config_tcn_default.yaml')) as file:
    default_config = yaml.safe_load(file)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1.squeeze()).unsqueeze(dim=2)


def train_model(model, optimizer, loss_fn, train_loader, epochs):
    """Training loop for TCN model."""

    model.train()
    train_loss, val_loss, = [], []
    with tqdm(range(epochs), unit="epoch", desc=f"Training TCN model") as pbar:
        for epoch in pbar:

            running_loss, total_cases, = 0, 0  # Running totals
            for seq, target in train_loader:
                seq, target = seq.type(torch.float32).to(device), target.type(torch.float32).to(device)

                # Forward backward
                outputs = model(seq)
                loss = loss_fn(target, outputs)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                running_loss += loss.item()
                total_cases += len(seq)
            train_loss.append(running_loss / total_cases)
            pbar.set_postfix(train_loss=train_loss[epoch])
    plt.plot(train_loss, 'b', label='Training loss')
    plt.title('Loss history')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_forecasts(model, dataloader):
    x, y, yhat = [], [], []

    model.eval()
    with torch.no_grad():
        for seq, target in dataloader:
            seq, target = seq.type(torch.float32).to(device), target.type(torch.float32).to(device)

            x.extend(seq.squeeze().cpu().numpy())
            y.extend(target.squeeze().cpu().numpy())
            yhat.extend(model(seq).squeeze().cpu().numpy())
    return np.array(x), np.array(y), np.array(yhat)


def run_tcn_model(train_dl, test_dl, in_size, out_size, epochs=30, nn_config=default_config):

    model = TCN(in_size, out_size, nn_config['channel_sizes'], nn_config['kernel_size'], nn_config['dropout'])

    optimizer = torch.optim.Adam(model.parameters(), lr=nn_config['learning_rate'])
    loss_fn = torch.nn.MSELoss()

    train_model(model, optimizer, loss_fn, train_dl, epochs=epochs)

    # Get forecasts on test
    x_test, y_test, yhat_test = get_forecasts(model, test_dl)

    return x_test, y_test, yhat_test
