import random

import pennylane as qml
import tensorflow as tf
import torch
from pennylane import numpy as np
from sklearn.decomposition import PCA
from torch import nn

dev = qml.device('default.qubit', wires=4)

def data_load_and_process():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[
        ..., np.newaxis] / 255.0
    train_filter_tf = np.where((y_train == 0) | (y_train == 1))
    test_filter_tf = np.where((y_test == 0) | (y_test == 1))

    x_train, y_train = x_train[train_filter_tf], y_train[train_filter_tf]
    x_test, y_test = x_test[test_filter_tf], y_test[test_filter_tf]

    x_train = tf.image.resize(x_train[:], (256, 1)).numpy()
    x_test = tf.image.resize(x_test[:], (256, 1)).numpy()
    x_train, x_test = tf.squeeze(x_train).numpy(), tf.squeeze(x_test).numpy()

    X_train = PCA(4).fit_transform(x_train)
    X_test = PCA(4).fit_transform(x_test)
    x_train, x_test = [], []
    for x in X_train:
        x = (x - x.min()) * (np.pi / (x.max() - x.min()))
        x_train.append(x)
    for x in X_test:
        x = (x - x.min()) * (np.pi / (x.max() - x.min()))
        x_test.append(x)
    return x_train[:400], x_test[:100], y_train[:400], y_test[:100]


# make new data for hybrid model
def new_data(batch_size, X, Y):
    X1_new, X2_new, Y_new = [], [], []
    for i in range(batch_size):
        n, m = np.random.randint(len(X)), np.random.randint(len(X))
        X1_new.append(X[n])
        X2_new.append(X[m])
        if Y[n] == Y[m]:
            Y_new.append(1)
        else:
            Y_new.append(0)
    X1_new, X2_new, Y_new = torch.tensor(X1_new).to(
        torch.float32), torch.tensor(X2_new).to(torch.float32), torch.tensor(
        Y_new).to(torch.float32)
    return X1_new, X2_new, Y_new




# Define the quantum circuit with dynamic actions
def QuantumEmbedding(inputs, action):
    if action == 0:  # RX applied to all qubits
        for i in range(4):
            qml.RX(inputs[i], wires=i)
    elif action == 1:  # RY applied to all qubits
        for i in range(4):
            qml.RY(inputs[i], wires=i)
    elif action == 2:  # RZ applied to all qubits
        for i in range(4):
            qml.RZ(inputs[i], wires=i)
    elif action == 3:  # CNOT in linear nearest-neighbor configuration
        for i in range(3):  # CNOT from qubit i to i+1
            qml.CNOT(wires=[i, i + 1])


@qml.qnode(dev, interface="torch")
def circuit(action, inputs):
    QuantumEmbedding(inputs[0:4], action)
    qml.adjoint(QuantumEmbedding)(inputs[4:8], action)
    return qml.probs(wires=range(4))


# Simple RL agent using Policy Gradient (REINFORCE)
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs



if __name__ == "__main__":
    gamma = 0.9
    learning_rate = 0.01

    action_set = ['RX', 'RY', 'RZ', 'CNOT']
    state_size = 4
    action_size = len(action_set)
    policy = PolicyNetwork(state_size, action_size)
    optimizer = qml.NesterovMomentumOptimizer(stepsize=learning_rate)

    # load data
    X_train, X_test, Y_train, Y_test = data_load_and_process()

    batch_size = 25
    iterations = 7
    episodes = 6

    for episode in range(episodes):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        state = get_initial_state()
        actions = []
        log_probs = []
        rewards = []

        for it in range(iterations):
            action_probs = policy(state_tensor)



    torch.save(model.state_dict(), "model.pt")