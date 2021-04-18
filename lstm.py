import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import CaltransTraffic, split_dataset, train_val_dataset


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size), torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def main():
    window = 20
    horizon = 5

    traffic = CaltransTraffic("./data/mvdata/traffic.txt", 0, window, horizon, None)

    train, test = train_val_dataset(traffic, 0.2)

    train_loader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=16, shuffle=False, drop_last=True)

    model = LSTM(input_size=1, output_size=horizon, hidden_layer_size=100)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 150

    for i in range(epochs):
        for seq, labels in train_loader:

            print(seq)
            print(labels)
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

if __name__ == "__main__":
    main()

