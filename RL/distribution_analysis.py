import pennylane as qml
import torch
from pennylane import numpy as np
from torch import nn
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
                qml.RX(x[qubit_idx], wires=qubit_idx)
            elif action[qubit_idx] == 2:
                qml.RY(x[qubit_idx], wires=qubit_idx)
            elif action[qubit_idx] == 3:
                qml.RZ(x[qubit_idx], wires=qubit_idx)
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
    def __init__(self, tick=None, action_sequence=None,):
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
        elif it == (NQE_iterations-1):
            final_loss = loss.item()
        NQE_opt.zero_grad()
        loss.backward()
        NQE_opt.step()

        if it % 3 == 0:
            print(f"Iterations: {it} Loss: {loss.item()}")
    return initial_loss, final_loss


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




# Main iterative process
if __name__ == "__main__":
    # Parameter settings
    data_size = 4  # Data reduction size from 256->, determine # of qubit
    batch_size = 25

    # Parameter for NQE
    N_layers = 1
    NQE_iterations = 50

    # Load data
    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='mnist',
                                                             reduction_sz=data_size)

    action_sequence = [[random.choice(range(5)) for _ in range(4)] for _ in range(15)]

    # action_sequence.insert(0, [0, 0, 0, 0])
    # action_sequence.insert(0, [4, 4, 4, 4])
    # action_sequence.insert(0, [0, 0, 0, 0])
    # action_sequence.pop()
    # action_sequence.pop()
    # action_sequence.pop()

    # Step 1: Train NQE
    initial_loss, final_loss = train_NQE(X_train, Y_train, NQE_iterations,
                                      batch_size, 'rl', action_sequence)

    print('dd')

