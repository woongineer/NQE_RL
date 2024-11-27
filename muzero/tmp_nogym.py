import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from pennylane import numpy as pnp
from collections import deque
import random
import math

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Number of features and qubits
num_features = 4  # As specified
n_qubits = num_features

# Generate synthetic data for the problem
N = 400  # Total number of data samples
data = np.random.randn(N, num_features)
labels = np.random.randint(0, 2, size=N)  # Labels are 0 or 1

# Prepare data in triples [x1, x2, y], where y = 1 if labels are the same, else 0
triples = []
for _ in range(N):
    idx1 = np.random.randint(0, N)
    idx2 = np.random.randint(0, N)
    x1 = data[idx1]
    x2 = data[idx2]
    y1 = labels[idx1]
    y2 = labels[idx2]
    y = 1 if y1 == y2 else 0
    triples.append((x1, x2, y))

# Create mini-batches
batch_size = 25
num_batches = len(triples) // batch_size


# Define the action set as per the paper
# Each action corresponds to adding a specific gate layer to the circuit

# Action functions
def apply_X_layer():
    def layer(qc, x):
        for q in range(n_qubits):
            qc.PauliX(wires=q)

    return layer


def apply_Y_layer():
    def layer(qc, x):
        for q in range(n_qubits):
            qc.PauliY(wires=q)

    return layer


def apply_Z_layer():
    def layer(qc, x):
        for q in range(n_qubits):
            qc.PauliZ(wires=q)

    return layer


def apply_H_layer():
    def layer(qc, x):
        for q in range(n_qubits):
            qc.Hadamard(wires=q)

    return layer


def apply_CX_layer():
    def layer(qc, x):
        for q in range(n_qubits - 1):
            qc.CNOT(wires=[q, q + 1])

    return layer


def apply_CY_layer():
    def layer(qc, x):
        for q in range(n_qubits - 1):
            qc.CY(wires=[q, q + 1])

    return layer


def apply_CZ_layer():
    def layer(qc, x):
        for q in range(n_qubits - 1):
            qc.CZ(wires=[q, q + 1])

    return layer


def apply_CRx_layer(n):
    def layer(qc, x):
        angle = np.pi / n
        for q in range(n_qubits - 1):
            qc.CRX(angle, wires=[q, q + 1])

    return layer


def apply_CRy_layer(n):
    def layer(qc, x):
        angle = np.pi / n
        for q in range(n_qubits - 1):
            qc.CRY(angle, wires=[q, q + 1])

    return layer


def apply_CRz_layer(n):
    def layer(qc, x):
        angle = np.pi / n
        for q in range(n_qubits - 1):
            qc.CRZ(angle, wires=[q, q + 1])

    return layer


def apply_Rx_pi_x_layer():
    def layer(qc, x):
        for q in range(n_qubits):
            qc.RX(np.pi * x[q], wires=q)

    return layer


def apply_Ry_pi_x_layer():
    def layer(qc, x):
        for q in range(n_qubits):
            qc.RY(np.pi * x[q], wires=q)

    return layer


def apply_Rz_pi_x_layer():
    def layer(qc, x):
        for q in range(n_qubits):
            qc.RZ(np.pi * x[q], wires=q)

    return layer


def apply_Rx_fixed_layer(n):
    def layer(qc, x):
        angle = np.pi / n
        for q in range(n_qubits):
            qc.RX(angle, wires=q)

    return layer


def apply_Ry_fixed_layer(n):
    def layer(qc, x):
        angle = np.pi / n
        for q in range(n_qubits):
            qc.RY(angle, wires=q)

    return layer


def apply_Rz_fixed_layer(n):
    def layer(qc, x):
        angle = np.pi / n
        for q in range(n_qubits):
            qc.RZ(angle, wires=q)

    return layer


def apply_Rx_arctan_layer():
    def layer(qc, x):
        for q in range(n_qubits):
            qc.RX(np.arctan(x[q]), wires=q)

    return layer


def apply_Ry_arctan_layer():
    def layer(qc, x):
        for q in range(n_qubits):
            qc.RY(np.arctan(x[q]), wires=q)

    return layer


def apply_Rz_arctan_layer():
    def layer(qc, x):
        for q in range(n_qubits):
            qc.RZ(np.arctan(x[q]), wires=q)

    return layer


# Build the action set
action_set = {}
action_id = 0

# Single-qubit Pauli gates
action_set[action_id] = apply_X_layer()
action_id += 1
action_set[action_id] = apply_Y_layer()
action_id += 1
action_set[action_id] = apply_Z_layer()
action_id += 1

# Hadamard gate
action_set[action_id] = apply_H_layer()
action_id += 1

# Controlled gates
action_set[action_id] = apply_CX_layer()
action_id += 1
action_set[action_id] = apply_CY_layer()
action_id += 1
action_set[action_id] = apply_CZ_layer()
action_id += 1

# Controlled rotations with n ∈ {1,2,3,4,8}
for n in [1, 2, 3, 4, 8]:
    action_set[action_id] = apply_CRx_layer(n)
    action_id += 1
for n in [1, 2, 3, 4, 8]:
    action_set[action_id] = apply_CRy_layer(n)
    action_id += 1
for n in [1, 2, 3, 4, 8]:
    action_set[action_id] = apply_CRz_layer(n)
    action_id += 1

# Rotations encoding data
action_set[action_id] = apply_Rx_pi_x_layer()
action_id += 1
action_set[action_id] = apply_Ry_pi_x_layer()
action_id += 1
action_set[action_id] = apply_Rz_pi_x_layer()
action_id += 1

# Rotations with fixed angles
for n in [1, 2, 3, 4, 8]:
    action_set[action_id] = apply_Rx_fixed_layer(n)
    action_id += 1
for n in [1, 2, 3, 4, 8]:
    action_set[action_id] = apply_Ry_fixed_layer(n)
    action_id += 1
for n in [1, 2, 3, 4, 8]:
    action_set[action_id] = apply_Rz_fixed_layer(n)
    action_id += 1

# Nonlinear rotations
action_set[action_id] = apply_Rx_arctan_layer()
action_id += 1
action_set[action_id] = apply_Ry_arctan_layer()
action_id += 1
action_set[action_id] = apply_Rz_arctan_layer()
action_id += 1

num_actions = action_id


# Define the environment for MuZero
class QuantumCircuitEnv:
    def __init__(self, action_set, max_depth=5):
        self.action_set = action_set
        self.max_depth = max_depth  # Maximum circuit depth
        self.reset()

    def reset(self):
        self.actions = []  # List of actions taken (gate indices)
        self.depth = 0  # Current circuit depth
        return self.get_observation()

    def step(self, action):
        self.actions.append(action)
        self.depth += 1
        done = self.depth >= self.max_depth
        observation = self.get_observation()
        reward = 0  # Reward will be computed externally
        return observation, reward, done, {}

    def get_observation(self):
        # The observation is the current sequence of actions (circuit configuration)
        # For the MuZero representation function, we convert it to a fixed-size tensor
        obs = torch.zeros(self.max_depth, dtype=torch.long)
        obs[:len(self.actions)] = torch.tensor(self.actions, dtype=torch.long)
        return obs

    def get_circuit(self):
        # Return the current sequence of actions (circuit)
        return self.actions


# Define the MuZero components
class MuZeroNet(nn.Module):
    def __init__(self, observation_shape, action_space_size, hidden_size=128):
        super(MuZeroNet, self).__init__()
        self.hidden_size = hidden_size
        self.action_space_size = action_space_size

        # Representation function (h)
        self.representation_net = nn.Sequential(
            nn.Embedding(num_actions, hidden_size),
            nn.LSTM(hidden_size, hidden_size, batch_first=True),
            nn.Flatten(),
            nn.Linear(hidden_size * observation_shape[0], hidden_size),
            nn.ReLU()
        )

        # Dynamics function (g)
        self.dynamics_net = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),  # +1 for action
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Prediction function (f)
        self.policy_net = nn.Linear(hidden_size, action_space_size)
        self.value_net = nn.Linear(hidden_size, 1)

    def initial_inference(self, observation):
        batch_size = observation.shape[0]
        hidden_state = self.representation_net(observation)
        policy_logits = self.policy_net(hidden_state)
        value = self.value_net(hidden_state)
        return value, policy_logits, hidden_state

    def recurrent_inference(self, hidden_state, action):
        # Convert action to one-hot encoding
        action_onehot = torch.zeros((action.shape[0], 1), dtype=torch.float32)
        action_onehot[:,
        0] = action.float() / self.action_space_size  # Normalize action
        x = torch.cat([hidden_state, action_onehot], dim=1)
        next_hidden_state = self.dynamics_net(x)
        policy_logits = self.policy_net(next_hidden_state)
        value = self.value_net(next_hidden_state)
        return value, policy_logits, next_hidden_state


# Monte Carlo Tree Node
class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = 0  # Not used in this context
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


# MCTS implementation for MuZero
def mcts(config, root, action_set, muzero_net, env, batch, num_simulations):
    for _ in range(num_simulations):
        node = root
        search_path = [node]
        current_env = QuantumCircuitEnv(action_set, max_depth=env.max_depth)
        current_env.actions = env.actions.copy()

        # Traverse the tree
        while node.expanded():
            action, node = select_child(node)
            current_env.step(action)
            search_path.append(node)

        # Expand leaf node
        observation = current_env.get_observation().unsqueeze(0)
        value, policy_logits, hidden_state = muzero_net.initial_inference(
            observation)
        value = value.item()
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).detach().numpy()

        node.hidden_state = hidden_state
        expand_node(node, action_space_size, policy)

        # Evaluate the leaf node
        reward = -compute_batch_loss(batch, current_env.get_circuit()).item()

        # Backpropagate the value and reward
        backpropagate(search_path, value, reward)

    # Choose the action with the highest visit count
    max_visit_count = -1
    best_action = None
    for action, child in root.children.items():
        if child.visit_count > max_visit_count:
            max_visit_count = child.visit_count
            best_action = action
    return best_action


def select_child(node):
    # UCB formula
    max_ucb = -float('inf')
    best_action = None
    best_child = None
    for action, child in node.children.items():
        ucb_score = ucb(child, node)
        if ucb_score > max_ucb:
            max_ucb = ucb_score
            best_action = action
            best_child = child
    return best_action, best_child


def ucb(child, parent, c1=1.25, c2=19652):
    prior_score = child.prior * math.sqrt(parent.visit_count) / (
                1 + child.visit_count)
    value_score = child.value()
    return value_score + prior_score


def expand_node(node, action_space_size, policy):
    for action in range(action_space_size):
        node.children[action] = Node(policy[action])


def backpropagate(search_path, value, reward):
    for node in reversed(search_path):
        node.value_sum += value
        node.visit_count += 1


# Compute batch loss based on the custom loss function
def compute_batch_loss(batch, actions):
    x1_batch = np.array([item[0] for item in batch])
    x2_batch = np.array([item[1] for item in batch])
    y_batch = np.array([item[2] for item in batch])

    batch_size = len(batch)

    state_i_batch = get_state_batch(x1_batch, actions)
    state_j_batch = get_state_batch(x2_batch, actions)

    inner_products = np.einsum('bi,bi->b', state_i_batch.conj(), state_j_batch)
    fidelities = np.abs(inner_products) ** 2

    targets = 0.5 * (1 + y_batch)
    losses = (fidelities - targets) ** 2

    total_loss = np.mean(losses)
    return torch.tensor(total_loss, dtype=torch.float32)


# Function to build and run the quantum circuit for a batch of data
def get_state_batch(x_batch, actions):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev, interface='numpy')
    def circuit(x):
        for action_id in actions:
            layer = action_set[action_id]
            layer(qml, x)
        return qml.state()

    state_batch = []
    for x in x_batch:
        state = circuit(x)
        state_batch.append(state)
    return np.array(state_batch)


# Training loop
def train_muzero(env, muzero_net, optimizer, data_loader, num_episodes=100):
    for episode in range(num_episodes):
        env.reset()
        done = False
        total_loss = 0

        # Get the initial observation
        observation = env.get_observation().unsqueeze(0)
        root_value, policy_logits, hidden_state = muzero_net.initial_inference(
            observation)
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).detach().numpy()
        root = Node(0)
        expand_node(root, action_space_size, policy)
        root.hidden_state = hidden_state

        while not done:
            # Get the next batch from the data loader
            try:
                batch = next(data_loader)
            except StopIteration:
                data_loader = iter(train_loader)
                batch = next(data_loader)

            # Run MCTS to get the next action
            action = mcts(config={}, root=root, action_set=action_set,
                          muzero_net=muzero_net, env=env, batch=batch,
                          num_simulations=25)

            # Take action in the environment
            next_observation, _, done, _ = env.step(action)
            next_observation = next_observation.unsqueeze(0)

            # Compute reward based on the custom loss function
            reward = -compute_batch_loss(batch, env.get_circuit()).item()

            # Update the network
            # Recurrent inference
            action_tensor = torch.tensor([action], dtype=torch.float32)
            value, policy_logits, hidden_state = muzero_net.recurrent_inference(
                root.hidden_state, action_tensor)
            policy = torch.softmax(policy_logits, dim=1).squeeze(
                0).detach().numpy()

            # Create a new root node
            root = Node(0)
            expand_node(root, action_space_size, policy)
            root.hidden_state = hidden_state

            # Compute losses
            target_value = torch.tensor([[reward]], dtype=torch.float32)
            value_loss = (value - target_value.detach()).pow(2).mean()
            policy_target = torch.zeros((1, muzero_net.action_space_size))
            policy_target[0, action] = 1.0
            policy_loss = nn.CrossEntropyLoss()(policy_logits,
                                                torch.tensor([action]))
            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Loss: {total_loss:.4f}")

    print("Training completed.")


# Instantiate the environment and MuZero network
env = QuantumCircuitEnv(action_set, max_depth=5)
observation_shape = (env.max_depth,)
action_space_size = num_actions
muzero_net = MuZeroNet(observation_shape, action_space_size)
optimizer = optim.Adam(muzero_net.parameters(), lr=1e-3)

# Prepare data loader
random.shuffle(triples)
batches = [triples[i:i + batch_size] for i in
           range(0, len(triples), batch_size)]
train_loader = iter(batches)

# Train MuZero
train_muzero(env, muzero_net, optimizer, train_loader, num_episodes=100)

# After training, you can retrieve the best circuit
best_actions = env.get_circuit()
print("Best action sequence:", best_actions)
