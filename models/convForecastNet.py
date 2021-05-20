"""
ForecastNet with cells comprising a convolutional neural network.
ForecastNetConvModel provides the mixture density network outputs.
ForecastNetConvModel2 provides the linear outputs.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
import time
from dataHelpers import format_input
from gaussian import gaussian_loss, mse_loss
import torch.nn.functional as F



class ForecastNetConvModel2(nn.Module):
    """
    Class for the convolutional hidden cell version of the model
    """
    def __init__(self, input_dim, hidden_dim, output_dim, in_seq_length, out_seq_length, device):
        """
        Constructor
        :param input_dim: Dimension of the inputs
        :param hidden_dim: Number of hidden units
        :param output_dim: Dimension of the outputs
        :param in_seq_length: Length of the input sequence
        :param out_seq_length: Length of the output sequence
        :param device: The device on which compuations are perfomed.
        """
        super(ForecastNetConvModel2, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.device = device

        self.conv_layer1 = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=5, padding=2) for i in range(out_seq_length)])
        self.conv_layer2 = nn.ModuleList([nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1) for i in range(out_seq_length)])
        flatten_layer = [nn.Linear(hidden_dim * (input_dim * in_seq_length), hidden_dim)]
        for i in range(out_seq_length - 1):
            flatten_layer.append(nn.Linear(hidden_dim * (input_dim * in_seq_length + hidden_dim + output_dim), hidden_dim))
        self.flatten_layer = nn.ModuleList(flatten_layer)
        self.output_layer = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(out_seq_length)])

        # # Convolutional Layers with Pooling
        # self.conv_layer1 = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=5, padding=2) for i in range(out_seq_length)])
        # self.pool_layer1 = nn.ModuleList([nn.AvgPool1d(kernel_size=2, padding=0) for i in range(out_seq_length)])
        # self.conv_layer2 = nn.ModuleList([nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1) for i in range(out_seq_length)])
        # self.pool_layer2 = nn.ModuleList([nn.AvgPool1d(kernel_size=2, padding=0) for i in range(out_seq_length) for i in range(out_seq_length)])
        # flatten_layer = [nn.Linear(hidden_dim//4 * (input_dim * in_seq_length), hidden_dim)]
        # for i in range(out_seq_length - 1):
        #     flatten_layer.append(nn.Linear(hidden_dim * ((input_dim * in_seq_length + hidden_dim + output_dim) // 4), hidden_dim))
        # self.flatten_layer = nn.ModuleList(flatten_layer)
        # self.output_layer = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(out_seq_length)])

    def forward(self, input, target, is_training=False):
        """
        Forward propagation of the convolutional ForecastNet model
        :param input: Input data in the form [input_seq_length, batch_size, input_dim]
        :param target: Target data in the form [output_seq_length, batch_size, output_dim]
        :param is_training: If true, use target data for training, else use the previous output.
        :return: outputs: Forecast outputs in the form [decoder_seq_length, batch_size, input_dim]
        """
        # Initialise outputs
        outputs = torch.zeros((self.out_seq_length, input.shape[0], self.output_dim)).to(self.device)
        # First input
        next_cell_input = input.unsqueeze(dim=1)
        # Propagate through network
        for i in range(self.out_seq_length):
            # Propagate through the cell
            hidden = F.relu(self.conv_layer1[i](next_cell_input))
            # hidden = self.pool_layer1[i](hidden)
            hidden = F.relu(self.conv_layer2[i](hidden))
            # hidden = self.pool_layer2[i](hidden)
            hidden = hidden.reshape((input.shape[0], -1))
            hidden = F.relu(self.flatten_layer[i](hidden))

            # Calculate output
            output = self.output_layer[i](hidden)
            outputs[i,:,:] = output

            # Prepare the next input
            if is_training:
                next_cell_input = torch.cat((input, hidden, target[i, :, :]), dim=1).unsqueeze(dim=1)
            else:
                next_cell_input = torch.cat((input, hidden, outputs[i, :, :]), dim=1).unsqueeze(dim=1)
            # Concatenate next input and
        return outputs


def format_input(input):
    """
    Format the input array by combining the time and input dimension of the input for feeding into ForecastNet.
    That is: reshape from [in_seq_length, n_batches, input_dim] to [n_batches, in_seq_length * input_dim]
    :param input: Input tensor with shape [in_seq_length, n_batches, input_dim]
    :return: input tensor reshaped to [n_batches, in_seq_length * input_dim]
    """
    in_seq_length, batch_size, input_dim = input.shape
    input_reshaped = input.permute(1, 0, 2)
    input_reshaped = torch.reshape(input_reshaped, (batch_size, -1))
    return input_reshaped

# Set plot_train_progress to True if you want a plot of the forecast after each epoch
plot_train_progress = False
if plot_train_progress:
    import matplotlib.pyplot as plt


# Set plot_train_progress to True if you want a plot of the forecast after each epoch
plot_train_progress = False
if plot_train_progress:
    import matplotlib.pyplot as plt


def train(fcstnet, train_x, train_y, validation_x=None, validation_y=None, restore_session=False):
    """
    Train the ForecastNet model on a provided dataset.
    In the following variable descriptions, the input_seq_length is the length of the input sequence
    (2*seasonal_period in the paper) and output_seq_length is the number of steps-ahead to forecast
    (seasonal_period in the paper). The n_batches is the total number batches in the dataset. The
    input_dim and output_dim are the dimensions of the input sequence and output sequence respectively
    (in the paper univariate sequences were used where input_dim=output_dim=1).
    :param fcstnet: A forecastNet object defined by the class in forecastNet.py.
    :param train_x: Input training data in the form [input_seq_length, n_batches, input_dim]
    :param train_y: Target training data in the form [output_seq_length, n_batches, output_dim]
    :param validation_x: Optional input validation data in the form [input_seq_length, n_batches, input_dim]
    :param validation_y: Optional target validation data in the form [output_seq_length, n_batches, output_dim]
    :param restore_session: If true, restore parameters and keep training, else train from scratch
    :return: training_costs: a list of training costs over the set of epochs
    :return: validation_costs: a list of validation costs over the set of epochs
    """

    # Convert numpy arrays to Torch tensors
    if type(train_x) is np.ndarray:
        train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    if type(train_y) is np.ndarray:
        train_y = torch.from_numpy(train_y).type(torch.FloatTensor)
    if type(validation_x) is np.ndarray:
        validation_x = torch.from_numpy(validation_x).type(torch.FloatTensor)
    if type(validation_y) is np.ndarray:
        validation_y = torch.from_numpy(validation_y).type(torch.FloatTensor)

    # Format inputs
    train_x = format_input(train_x)
    validation_x = format_input(validation_x)

    validation_x = validation_x.to(fcstnet.device)
    validation_y = validation_y.to(fcstnet.device)

    # Initialise model with predefined parameters
    if restore_session:
        # Load model parameters
        checkpoint = torch.load(fcstnet.save_file)
        fcstnet.model.load_state_dict(checkpoint['model_state_dict'])
        fcstnet.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Number of batch samples
    n_samples = train_x.shape[0]

    # List to hold the training costs over each epoch
    training_costs = []
    validation_costs = []

    # Set in training mode
    fcstnet.model.train()

    # Training loop
    for epoch in range(fcstnet.n_epochs):

        # Start the epoch timer
        t_start = time.time()

        # Print the epoch number
        print('Epoch: %i of %i' % (epoch + 1, fcstnet.n_epochs))

        # Initial average epoch cost over the sequence
        batch_cost = []

        # Counter for permutation loop
        count = 0

        # Permutation to randomly sample from the dataset
        permutation = np.random.permutation(np.arange(0, n_samples, fcstnet.batch_size))

        # Loop over the permuted indexes, extract a sample at that index and run it through the model
        for sample in permutation:
            # Extract a sample at the current permuted index
            input = train_x[sample:sample + fcstnet.batch_size, :]
            target = train_y[:, sample:sample + fcstnet.batch_size, :]

            # Send input and output data to the GPU/CPU
            input = input.to(fcstnet.device)
            target = target.to(fcstnet.device)

            # Zero the gradients
            fcstnet.optimizer.zero_grad()
            # Calculate the outputs
            outputs = fcstnet.model(input, target, is_training=True)
            loss = F.mse_loss(input=outputs, target=target)
            batch_cost.append(loss.item())
            # Calculate the derivatives
            loss.backward()
            # Update the model parameters
            fcstnet.optimizer.step()

            if count % 50 == 0:
                print("Average cost after training batch %i of %i: %f" % (count, permutation.shape[0], loss.item()))
            count += 1
        # Find average cost over sequences and batches
        epoch_cost = np.mean(batch_cost)
        # Calculate the average training cost over the sequence
        training_costs.append(epoch_cost)

        # Plot an animation of the training progress
        if plot_train_progress:
            plt.cla()
            plt.plot(np.arange(input.shape[0], input.shape[0] + target.shape[0]), target[:, 0, 0])
            temp = outputs.detach()
            plt.plot(np.arange(input.shape[0], input.shape[0] + target.shape[0]), temp[:, 0, 0])
            plt.pause(0.1)

        # Validation tests
        if validation_x is not None:
            fcstnet.model.eval()
            with torch.no_grad():
                # Compute outputs and loss for a mixture density network output
                if fcstnet.model_type == 'dense' or fcstnet.model_type == 'conv':
                    # Calculate the outputs
                    y_valid, mu_valid, sigma_valid = fcstnet.model(validation_x, validation_y, is_training=False)
                    # Calculate the loss
                    loss = gaussian_loss(z=validation_y, mu=mu_valid, sigma=sigma_valid)
                # Compute outputs and loss for a linear output
                elif fcstnet.model_type == 'dense2' or fcstnet.model_type == 'conv2':
                    # Calculate the outputs
                    y_valid = fcstnet.model(validation_x, validation_y, is_training=False)
                    # Calculate the loss
                    loss = F.mse_loss(input=y_valid, target=validation_y)
                validation_costs.append(loss.item())
            fcstnet.model.train()

        # Print progress
        print("Average epoch training cost: ", epoch_cost)
        if validation_x is not None:
            print('Average validation cost:     ', validation_costs[-1])
        print("Epoch time:                   %f seconds" % (time.time() - t_start))
        print("Estimated time to complete:   %.2f minutes, (%.2f seconds)" %
              ((fcstnet.n_epochs - epoch - 1) * (time.time() - t_start) / 60,
               (fcstnet.n_epochs - epoch - 1) * (time.time() - t_start)))

        # Save a model checkpoint
        best_result = False
        if validation_x is None:
            if training_costs[-1] == min(training_costs):
                best_result = True
        else:
            if validation_costs[-1] == min(validation_costs):
                best_result = True
        if best_result:
            torch.save({
                'model_state_dict': fcstnet.model.state_dict(),
                'optimizer_state_dict': fcstnet.optimizer.state_dict(),
            }, fcstnet.save_file)
            print("Model saved in path: %s" % fcstnet.save_file)

    return training_costs, validation_costs
