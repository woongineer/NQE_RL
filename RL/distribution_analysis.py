import pennylane as qml
import torch
from pennylane import numpy as np
from torch import nn
import pandas as pd
import random
import matplotlib.pyplot as plt

from data import new_data, data_load_and_process

# Set your device
dev = qml.device('default.qubit', wires=4)


def exp_Z(x, wires):
    qml.RZ(-2 * x, wires=wires)  ##TODO 왜 이걸 없애니까...???????


# exp(i(pi - x1)(pi - x2)ZZ) gate
def exp_ZZ2(x1, x2, wires):
    qml.CNOT(wires=wires)
    qml.RZ(-2 * (np.pi - x1) * (np.pi - x2), wires=wires[1])
    qml.CNOT(wires=wires)


def quantum_embedding_zz(input):
    for i in range(N_layers):
        for j in range(4):
            qml.Hadamard(wires=j)
            exp_Z(input[j], wires=j)
        for k in range(3):
            exp_ZZ2(input[k], input[k + 1], wires=[k, k + 1])
        exp_ZZ2(input[3], input[0], wires=[3, 0])


def quantum_embedding_rl(x, action_sequence):
    for action in action_sequence:
        for qubit_idx in range(data_size):
            if action[qubit_idx] == 0:
                qml.Hadamard(wires=qubit_idx)
            elif action[qubit_idx] == 1:
                qml.RX(-2*x[qubit_idx], wires=qubit_idx)
            elif action[qubit_idx] == 2:
                qml.RY(-2*x[qubit_idx], wires=qubit_idx)
            elif action[qubit_idx] == 3:
                qml.RZ(-2*x[qubit_idx], wires=qubit_idx)
            elif action[qubit_idx] == 4:
                qml.CNOT(wires=[qubit_idx, (qubit_idx + 1) % data_size])


def stater(strat, X1_batch, X2_batch, action_sequence=None):
    plt.clf()
    results = []
    for _ in range(30):
        NQE_model = NQEModel(tick=strat, action_sequence=action_sequence)
        NQE_model.train()
        results.append(NQE_model(X1_batch, X2_batch))
    results = torch.cat(results).flatten()
    print(torch.sum(results >= 0.96).item())
    plt.hist(results.detach().numpy(), bins=25, range=(0, 1))
    plt.savefig(f"{strat}.png")
    plt.clf()
    plt.close()


# Define the NQE Model
class NQEModel(torch.nn.Module):
    def __init__(self, tick=None, action_sequence=None, ):
        super().__init__()
        self.action_sequence = action_sequence
        if tick == 'zz':
            @qml.qnode(dev, interface="torch")
            def circuit(inputs):
                quantum_embedding_zz(inputs[0:4])
                qml.adjoint(quantum_embedding_zz)(inputs[4:8])
                return qml.probs(wires=range(4))

        elif tick == 'rl':
            @qml.qnode(dev, interface="torch")
            def circuit(inputs):
                quantum_embedding_rl(inputs[0:4], self.action_sequence)
                qml.adjoint(quantum_embedding_rl)(inputs[4:8],  ##TODO 되는거 맞나?
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


# Function to train NQE
def train_NQE(X_train, Y_train, NQE_iterations, batch_size, tick,
              action_sequence=None):
    # NQE_model = NQEModel(tick='zz')
    NQE_model = NQEModel(tick=tick, action_sequence=action_sequence)

    NQE_model.train()
    NQE_loss_fn = torch.nn.MSELoss()
    NQE_opt = torch.optim.SGD(NQE_model.parameters(), lr=0.01)

    for it in range(NQE_iterations):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        # stater('rl', X1_batch, X2_batch, action_sequence)
        # Y_batch.numpy().sum()
        pred = NQE_model(X1_batch, X2_batch)
        loss = NQE_loss_fn(pred, Y_batch)

        if it == 0:
            initial_loss = loss.item()
        elif it == (NQE_iterations - 1):
            final_loss = loss.item()
        NQE_opt.zero_grad()
        loss.backward()
        NQE_opt.step()

        if it % 3 == 0:
            print(f"Iterations: {it} Loss: {loss.item()}")
    return initial_loss, final_loss


def distribution_check(X_train, Y_train, batch_size, tick, action_sequence):
    NQE_model = NQEModel(tick=tick, action_sequence=action_sequence)

    NQE_model.train()
    NQE_loss_fn = torch.nn.MSELoss()

    X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
    pred = NQE_model(X1_batch, X2_batch)
    loss = NQE_loss_fn(pred, Y_batch)

    return loss.item()


# Function to transform data using NQE
def transform_data(NQE_model, X_data):
    NQE_model.eval()
    transformed_data = []
    with torch.no_grad():
        for x in X_data:
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            x_transformed = NQE_model.linear_relu_stack1(x_tensor)
            x_transformed = x_transformed.squeeze(0).detach().numpy()
            transformed_data.append(x_transformed)
    return transformed_data


def draw_circuit(depth, action_seq, minmax):
    @qml.qnode(dev)
    def fig_circ(action_seq):
        quantum_embedding_rl(np.array([1, 1, 1, 1]), action_seq)
        return qml.probs(wires=range(4))

    fig, ax = qml.draw_mpl(fig_circ)(action_seq)
    fig.text(0.5, 0.95, f"{minmax}", fontsize=14, ha='center', va='top')

    action_text = "\n".join(
        [str(action_seq[i:i + 5]) for i in
         range(0, len(action_seq), 5)]
    )
    fig.text(0.1, 0.1, f'{action_text}', fontsize=8, wrap=True)


    fig.savefig(f"dist_{depth}_{minmax}.png")

def draw_dist(loss, depth):
    plt.clf()
    plt.hist(loss, bins=100, range=(0, 1))

    min_loss = loss.min()
    max_loss = loss.max()
    median_loss = loss.median()
    mean_loss = loss.mean()
    variance_loss = loss.var()

    plt.text(0.02, plt.ylim()[1] * 0.9, f"Min: {min_loss:.4f}", fontsize=10)
    plt.text(0.02, plt.ylim()[1] * 0.85, f"Max: {max_loss:.4f}", fontsize=10)
    plt.text(0.02, plt.ylim()[1] * 0.8, f"Median: {median_loss:.4f}",
             fontsize=10)
    plt.text(0.02, plt.ylim()[1] * 0.75, f"Mean: {mean_loss:.4f}", fontsize=10)
    plt.text(0.02, plt.ylim()[1] * 0.7, f"Variance: {variance_loss:.4f}", fontsize=10)


    plt.savefig(f"dist_{depth}.png")


# Main iterative process
if __name__ == "__main__":
    # Parameter settings
    data_size = 4  # Data reduction size from 256->, determine # of qubit
    batch_size = 25

    # Parameter for NQE
    N_layers = 1
    NQE_iterations = 1

    # Load data
    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='mnist',
                                                             reduction_sz=data_size)

    for depth_elem in [3, 4, 5, 6, 7, 8]:
        print(f'{depth_elem} begin')
        depth = depth_elem
        dist_info = []
        for k in range(1000):
            print(f'{k}') if k % 20 == 0 else None
            action_sequence = [[random.choice(range(5)) for _ in range(4)] for _ in
                               range(depth)]
            loss = distribution_check(X_train, Y_train, batch_size, 'rl',
                                      action_sequence)
            dist_info.append({'action_sequence': action_sequence,
                              'loss': loss})

        dist_info_df = pd.DataFrame(dist_info)

        draw_dist(dist_info_df['loss'], depth)

        min_loss_row = dist_info_df.loc[dist_info_df['loss'].idxmin()]
        min_loss_action_sequence = min_loss_row['action_sequence']
        draw_circuit(depth, min_loss_action_sequence, 'min')

        max_loss_row = dist_info_df.loc[dist_info_df['loss'].idxmax()]
        max_loss_action_sequence = max_loss_row['action_sequence']
        draw_circuit(depth, max_loss_action_sequence, 'max')
