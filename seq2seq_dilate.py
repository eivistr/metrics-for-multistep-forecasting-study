import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_grulstm_layers, batch_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_grulstm_layers = num_grulstm_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers, batch_first=True)

    def forward(self, input_seq, hidden):  # input [batch_size, length T, dimensionality d]
        output, hidden = self.gru(input_seq, hidden)
        return output, hidden

    def init_hidden(self, device):
        # [num_layers*num_directions,batch,hidden_size]
        return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_grulstm_layers, fc_units, output_size):
        super(DecoderRNN, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, fc_units)
        self.out = nn.Linear(fc_units, output_size)

    def forward(self, input_seq, hidden):
        output, hidden = self.gru(input_seq, hidden)
        output = F.relu(self.fc(output))
        output = self.out(output)
        return output, hidden


class NetGRU(nn.Module):
    def __init__(self, encoder, decoder, target_length, device):
        super(NetGRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = target_length
        self.device = device

    def forward(self, x):
        input_length = x.shape[1]
        encoder_hidden = self.encoder.init_hidden(self.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[:, ei:ei + 1, :], encoder_hidden)

        decoder_input = x[:, -1, :].unsqueeze(1)  # first decoder input = last element of input sequence
        decoder_hidden = encoder_hidden

        outputs = torch.zeros([x.shape[0], self.target_length, x.shape[2]]).to(self.device)
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output
            outputs[:, di:di + 1, :] = decoder_output
        return outputs


def train_model(model, optimizer, loss_fn, train_loader, epochs):
    """Training loop for..."""

    train_loss, val_loss, = [], []
    with tqdm(range(epochs), unit="epoch", desc=f"DILATE GRU") as pbar:
        for epoch in pbar:

            model.train()  # Ensure training mode
            running_loss, total_cases, = 0, 0  # Running totals

            for i, (seq, target) in enumerate(train_loader, 0):
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
    plt.title('DILATE GRU loss history')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_forecasts(model, dataloader):
    x, y, yhat = [], [], []

    model.eval()  # Set evaluation mode
    with torch.no_grad():
        for seq, target in dataloader:
            seq, target = seq.type(torch.float32).to(device), target.type(torch.float32).to(device)

            x.extend(seq.squeeze().cpu().numpy())
            y.extend(target.squeeze().cpu().numpy())
            yhat.extend(model(seq).squeeze().cpu().numpy())
    return np.array(x), np.array(y), np.array(yhat)


# def predict_on_batch(model, test_loader, n):
#     batch_seq, batch_target = next(iter(test_loader))
#     batch_seq, batch_target = batch_seq.type(torch.float32).to(device), batch_target.type(torch.float32).to(device)
#
#     model.eval()
#     batch_forecast = gru_net(batch_seq)
#
#     for i in range(0, n):
#         seq = batch_seq.detach().cpu().numpy()[i, :, :]
#         target = batch_target.detach().cpu().numpy()[i, :, :]
#         forecast = batch_forecast.detach().cpu().numpy()[i, :, :]
#
#         plot_result(seq, target, forecast)
#         print("(TDI, TDM): ", tdi_tdm(target, forecast))