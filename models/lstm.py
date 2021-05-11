import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

with open(os.path.join(__location__, 'config_lstm_default.yaml')) as file:
    default_cfg = yaml.safe_load(file)


class EncoderLSTM(torch.nn.Module):
    def __init__(self, input_features, hidden_size, num_layers, dropout=0):
        super(EncoderLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                            batch_first=True)

    def forward(self, input_seq, hidden):  # input [batch_size, length T, dimensionality d]
        output, hidden = self.lstm(input_seq, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))


class DecoderLSTM(nn.Module):
    def __init__(self, input_features, output_features, hidden_size, num_layers, dropout=0):
        super(DecoderLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, int(hidden_size / 2))
        self.out = nn.Linear(int(hidden_size / 2), output_features)

    def forward(self, input_seq, hidden):
        output, hidden = self.lstm(input_seq, hidden)
        output = F.relu(self.fc(output))
        output = self.out(output)
        return output, hidden


class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder, target_length):
        super(Seq2SeqLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = target_length

        assert encoder.hidden_size == decoder.hidden_size, 'Hidden dimensions of encoder and decoder must be equal!'
        assert encoder.num_layers == decoder.num_layers, 'Encoder and decoder must have equal number of layers!'

    def forward(self, x):
        input_length = x.shape[1]
        batch_size = x.shape[0]
        encoder_hidden = self.encoder.init_hidden(batch_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[:, ei:ei + 1, :], encoder_hidden)

        decoder_input = x[:, -1, :].unsqueeze(1)  # first decoder input = last element of input sequence
        decoder_hidden = encoder_hidden

        outputs = torch.zeros([x.shape[0], self.target_length, x.shape[2]]).to(device)
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output
            outputs[:, di:di + 1, :] = decoder_output
        return outputs


def train_model(model, optimizer, loss_fn, train_loader, epochs, clip=0):
    model.train()
    train_loss, val_loss, = [], []
    with tqdm(range(epochs), unit="epoch", desc=f"Training LSTM model") as pbar:
        for epoch in pbar:

            running_loss, total_cases, = 0, 0  # Running totals
            for seq, target in train_loader:
                seq, target = seq.to(device), target.to(device)

                # Forward backward
                outputs = model(seq)
                loss = loss_fn(target, outputs)
                optimizer.zero_grad()
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
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
            seq, target = seq.to(device), target.to(device)

            x.extend(seq.squeeze().cpu().numpy())
            y.extend(target.squeeze().cpu().numpy())
            yhat.extend(model(seq).squeeze().cpu().numpy())

    x, y, yhat = np.array(x), np.array(y), np.array(yhat)
    assert x.shape == y.shape == yhat.shape, "Forecast outputs must have equal dimensions"

    return x, y, yhat


def run_lstm_model(train_dl, test_dl, out_size, epochs, cfg=default_cfg):
    encoder = EncoderLSTM(input_features=1, hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'],
                          dropout=cfg['dropout']).to(device)
    decoder = DecoderLSTM(input_features=1, hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'],
                          dropout=cfg['dropout'], output_features=1).to(device)
    model = Seq2SeqLSTM(encoder, decoder, out_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    loss_fn = torch.nn.MSELoss()

    train_model(model, optimizer, loss_fn, train_dl, epochs=epochs)

    # Get forecasts on test
    x_test, y_test, yhat_test = get_forecasts(model, test_dl)

    return x_test, y_test, yhat_test


if __name__ == '__main__':
    from datasets import get_dataset

    in_size = 24
    out_size = 24
    test_size = 500
    batch_size = 10
    epochs = 20
    lr = 0.01

    train, test = get_dataset('mvdata', 'traffic', in_size, out_size, test_size)
    train_dl = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

    run_lstm_model(train_dl, test_dl, out_size, epochs)