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
    """Training loop for autoencoder module."""

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







# def main():
#
#     # Parameters
#     batch_size = 32
#     window = 20
#     horizon = 20
#
#     traffic = CaltransTraffic("./datasets/mvdata/traffic.txt", 0, window, horizon, None)
#
#     train, test = train_test_dataset(traffic, 0.2)
#
#     train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
#     test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
#
#     encoder = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
#     decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, fc_units=16, output_size=1).to(device)
#     gru_net = NetGRU(encoder, decoder, horizon, device).to(device)
#
#     optimizer = torch.optim.Adam(gru_net.parameters(), lr=0.01)
#     loss_fn = torch.nn.MSELoss()
#
#     train_model(gru_net, optimizer, loss_fn, train_loader, epochs=10)
#
#     # Visualize results
#     test_seq, test_target = next(iter(test_loader))
#     test_seq, test_target = test_seq.type(torch.float32).to(device), test_target.type(torch.float32).to(device)
#
#     n = 10
#
#     gru_net.eval()
#
#     pred = gru_net(test_seq).to(device)
#     inputs = test_seq.detach().cpu().numpy()[1, :, :]
#     target = test_target.detach().cpu().numpy()[1, :, :]
#     preds = pred.detach().cpu().numpy()[1, :, :]
#
#     plt.plot(range(0, window), inputs, label='input', linewidth=3)
#     plt.plot(range(window-1, window+horizon), np.concatenate([inputs[window-1:window], target]), label='target', linewidth=3)
#     plt.plot(range(window-1, window+horizon),  np.concatenate([inputs[window-1:window], preds]), label='prediction', linewidth=3)
#     plt.xticks(range(0, 40, 2))
#     plt.legend()
#     plt.show()
#
#     # for i in range(1, n+1):
#     #     plt.figure()
#     #     plt.rcParams['figure.figsize'] = (17.0, 5.0)
#     #
#     #     pred = gru_net(test_seq).to(device)
#     #     inputs = test_seq.detach().cpu().numpy()[i, :, :]
#     #     target = test_target.detach().cpu().numpy()[i, :, :]
#     #     preds = pred.detach().cpu().numpy()[i, :, :]
#     #
#     #     plt.plot(range(0, window), inputs, label='input', linewidth=3)
#     #     plt.plot(range(window-1, window+horizon), np.concatenate([inputs[window-1:window], target]), label='target', linewidth=3)
#     #     plt.plot(range(window-1, window+horizon),  np.concatenate([inputs[window-1:window], preds]), label='prediction', linewidth=3)
#     #     plt.xticks(range(0, 40, 2))
#     #     plt.legend()
#     #     plt.show()


if __name__ == '__main__':
    main()