import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from pennylane import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf

# Define the quantum device
dev = qml.device('default.qubit', wires=4)

# Load and process data
def data_load_and_process():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
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

# Make new data for hybrid model
def new_data(batch_size, X, Y):
    X1_new, X2_new, Y_new = [], [], []
    for i in range(batch_size):
        n, m = np.random.randint(len(X)), np.random.randint(len(X))
        X1_new.append(X[n])
        X2_new.append(X[m])
        Y_new.append(1 if Y[n] == Y[m] else 0)
    return torch.tensor(X1_new).float(), torch.tensor(X2_new).float(), torch.tensor(Y_new).float()

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
    # Apply Quantum Embedding based on action
    QuantumEmbedding(inputs[0:4], action)
    # Apply adjoint of Quantum Embedding based on action
    qml.adjoint(QuantumEmbedding)(inputs[4:8], action)
    # Measure fidelity as probability of |00..0> state
    return qml.probs(wires=range(4))

# Policy Network using Policy Gradient (REINFORCE)
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs

def train_policy(policy, optimizer, rewards, log_probs):
    # Calculate the returns (discounted sum of rewards)
    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # Normalize returns for stability
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    # Calculate loss and update policy
    loss = 0
    for log_prob, R in zip(log_probs, returns):
        loss -= log_prob * R  # Policy Gradient loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    # Hyperparameters
    gamma = 0.98
    learning_rate = 0.01
    state_size = 8  # Input size for the policy (combined inputs)
    action_size = 4  # Number of possible actions
    policy = PolicyNetwork(state_size, action_size)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Load data
    X_train, X_test, Y_train, Y_test = data_load_and_process()
    batch_size = 25
    episodes = 10
    iterations = 7

    for episode in range(episodes):
        # Generate new data
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        state = torch.cat([X1_batch, X2_batch], dim=1)  # Combine states
        log_probs = []
        rewards = []

        for it in range(iterations):
            # Get action probabilities from the policy network
            action_probs = policy(state)
            m = Categorical(action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            log_probs.append(log_prob)

            # Apply the action to the quantum circuit and get the reward (fidelity)
            fidelity = circuit(action.item(), state)
            reward = fidelity.sum().item()  # Reward is the fidelity value
            rewards.append(reward)

            # The state remains the same because it's based on input data

        # Train the policy network using the rewards collected
        train_policy(policy, optimizer, rewards, log_probs)

        if episode % 2 == 0:
            print(f"Episode {episode}, Reward Sum: {sum(rewards)}")

    # Save the trained policy network
    torch.save(policy.state_dict(), "policy_model.pt")
