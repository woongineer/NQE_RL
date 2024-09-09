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
    QuantumEmbedding(inputs[0:4], action)
    qml.adjoint(QuantumEmbedding)(inputs[4:8], action)
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

def get_initial_state():
    # Initialize the state to represent the |0> state for all qubits
    # It could be a zero vector of length equal to the number of features describing the circuit
    return torch.zeros(state_size)

def update_state(current_state, action, inputs):
    # Update the state based on the action and inputs
    # For simplicity, we are using a placeholder approach here
    # In practice, you should define how to represent the updated state of the circuit
    # For example, this could be based on the outputs of the circuit
    # or the new parameters of the gates, etc.
    new_state = current_state.clone()  # Placeholder: Define a proper state update mechanism
    # Potentially update the state with information from the quantum measurement, etc.
    return new_state

if __name__ == "__main__":
    # Hyperparameters
    gamma = 0.98
    learning_rate = 0.01
    state_size = 8  # Input size for the policy (representing the circuit state)
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
        log_probs = []
        rewards = []

        # Initialize the circuit state
        current_state = get_initial_state()

        for it in range(iterations):
            # Forward pass to get action probabilities
            action_probs = policy(current_state)
            m = Categorical(action_probs)
            action = m.sample()  # Sample a single action

            # Get the log probability of the chosen action
            log_prob = m.log_prob(action)
            log_probs.append(log_prob)

            # Compute the reward (fidelity) for the chosen action
            fidelity = circuit(action.item(), torch.cat((X1_batch[it], X2_batch[it])).float())
            rewards.append(fidelity[0])  # Use fidelity as the reward

            # Update the state based on the action and the resulting circuit
            current_state = update_state(current_state, action, torch.cat((X1_batch[it], X2_batch[it])).float())

        # Train the policy network using the collected rewards and log_probs
        train_policy(policy, optimizer, rewards, log_probs)

        if episode % 2 == 0:
            print(f"Episode {episode}, Reward Sum: {sum(rewards)}")

    # Save the trained policy network
    torch.save(policy.state_dict(), "policy_model.pt")
