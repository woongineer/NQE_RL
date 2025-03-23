import random

import pandas as pd
import pennylane as qml
import torch
from torch import nn

from data import new_data, data_load_and_process

# Set your device
dev = qml.device('default.qubit', wires=4)


def quantum_embedding_rl(x, action_sequence):
    for action in action_sequence:
        for qubit_idx in range(data_size):
            if action[qubit_idx] == 0:
                qml.Hadamard(wires=qubit_idx)
            elif action[qubit_idx] == 1:
                qml.RX(-2 * x[qubit_idx], wires=qubit_idx)
            elif action[qubit_idx] == 2:
                qml.RY(-2 * x[qubit_idx], wires=qubit_idx)
            elif action[qubit_idx] == 3:
                qml.RZ(-2 * x[qubit_idx], wires=qubit_idx)
            elif action[qubit_idx] == 4:
                qml.CNOT(wires=[qubit_idx, (qubit_idx + 1) % data_size])


# Define the NQE Model
class NQEModel(torch.nn.Module):
    def __init__(self, action_sequence):
        super().__init__()
        self.action_sequence = action_sequence

        @qml.qnode(dev, interface="torch")
        def circuit(inputs):
            quantum_embedding_rl(inputs[0:4], self.action_sequence)
            qml.adjoint(quantum_embedding_rl)(inputs[4:8],
                                              self.action_sequence)
            return qml.probs(wires=range(4))

        self.qlayer1 = qml.qnn.TorchLayer(circuit, weight_shapes={})
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

    def forward(self, x1, x2):
        x1 = self.linear_relu_stack1(x1)
        x2 = self.linear_relu_stack1(x2)
        x = torch.concat([x1, x2], 1)
        x = self.qlayer1(x)
        return x[:, 0]


def distribution_check(X_train, Y_train, batch_size, action_sequence):
    NQE_model = NQEModel(action_sequence=action_sequence)

    NQE_model.train()
    NQE_loss_fn = torch.nn.MSELoss()

    X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
    pred = NQE_model(X1_batch, X2_batch)
    loss = NQE_loss_fn(pred, Y_batch)

    return loss.item()


# Main iterative process
if __name__ == "__main__":
    # Parameter settings
    data_size = 4
    batch_size = 100

    # Load data
    X_train, X_test, Y_train, _ = data_load_and_process(dataset='mnist',
                                                        reduction_sz=data_size)

    depth = 8
    dist_info = []
    for k in range(1000):
        print(f'{k}') if k % 20 == 0 else None
        action_sequence = [[random.choice(range(5)) for _ in range(4)] for _ in
                           range(depth)]
        loss = distribution_check(X_train, Y_train, batch_size, action_sequence)
        dist_info.append({'action_sequence': action_sequence, 'loss': loss})

    dist_info_df = pd.DataFrame(dist_info)

    check = 10
    new_df = dist_info_df.sort_values(by='loss')[:check]

    new_losses = []
    for act_seq in list(new_df['action_sequence']):
        for trial in range(10):
            new_loss = distribution_check(X_train, Y_train, batch_size, act_seq)
            new_losses.append({'action_sequence': act_seq, 'trial': trial,
                               'new_loss': new_loss})

    new_losses_df = pd.DataFrame(new_losses)

    new_df['action_sequence'] = new_df['action_sequence'].apply(str)
    new_losses_df['action_sequence'] = new_losses_df['action_sequence'].apply(
        str)
    merged_df = pd.merge(new_df, new_losses_df, on='action_sequence',
                         how='inner')
    merged_df['gap'] = merged_df['new_loss'] - merged_df['loss']
    merged_df.to_csv('min_batch_100.csv')







    new_df = dist_info_df.sort_values(by='loss').tail(check)

    new_losses = []
    for act_seq in list(new_df['action_sequence']):
        for trial in range(10):
            new_loss = distribution_check(X_train, Y_train, batch_size, act_seq)
            new_losses.append({'action_sequence': act_seq, 'trial': trial,
                               'new_loss': new_loss})

    new_losses_df = pd.DataFrame(new_losses)

    new_df['action_sequence'] = new_df['action_sequence'].apply(str)
    new_losses_df['action_sequence'] = new_losses_df['action_sequence'].apply(
        str)
    merged_df = pd.merge(new_df, new_losses_df, on='action_sequence',
                         how='inner')
    merged_df['gap'] = merged_df['new_loss'] - merged_df['loss']
    merged_df.to_csv('max_batch_100.csv')
