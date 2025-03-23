import random
from collections import deque
import math
import numpy as np
import pennylane as qml
import torch
from pennylane import numpy as np
from torch import nn

from data import data_load_and_process as dataprep
from data import new_data

# Set your device
n_qubit = 4
dev = qml.device('default.qubit', wires=n_qubit)

action_mapping = {
    0: 'X',
    1: 'Y',
    2: 'Z',
    3: 'CX',
    4: 'CY',
    5: 'CZ',
    # CRx(π/n)
    6: ('CRx_pi_over_n', 1),
    7: ('CRx_pi_over_n', 2),
    8: ('CRx_pi_over_n', 3),
    9: ('CRx_pi_over_n', 4),
    10: ('CRx_pi_over_n', 8),
    # CRy(π/n)
    11: ('CRy_pi_over_n', 1),
    12: ('CRy_pi_over_n', 2),
    13: ('CRy_pi_over_n', 3),
    14: ('CRy_pi_over_n', 4),
    15: ('CRy_pi_over_n', 8),
    # CRz(π/n)
    16: ('CRz_pi_over_n', 1),
    17: ('CRz_pi_over_n', 2),
    18: ('CRz_pi_over_n', 3),
    19: ('CRz_pi_over_n', 4),
    20: ('CRz_pi_over_n', 8),
    # Rx(πx), Ry(πx), Rz(πx)
    21: 'Rx_pi_x',
    22: 'Ry_pi_x',
    23: 'Rz_pi_x',
    # Rx(π/n)
    24: ('Rx_pi_over_n', 1),
    25: ('Rx_pi_over_n', 2),
    26: ('Rx_pi_over_n', 3),
    27: ('Rx_pi_over_n', 4),
    28: ('Rx_pi_over_n', 8),
    # Ry(π/n)
    29: ('Ry_pi_over_n', 1),
    30: ('Ry_pi_over_n', 2),
    31: ('Ry_pi_over_n', 3),
    32: ('Ry_pi_over_n', 4),
    33: ('Ry_pi_over_n', 8),
    # Rz(π/n)
    34: ('Rz_pi_over_n', 1),
    35: ('Rz_pi_over_n', 2),
    36: ('Rz_pi_over_n', 3),
    37: ('Rz_pi_over_n', 4),
    38: ('Rz_pi_over_n', 8),
    # Rx(arctan(x)), Ry(arctan(x)), Rz(arctan(x))
    39: 'Rx_arctan_x',
    40: 'Ry_arctan_x',
    41: 'Rz_arctan_x',
    42: 'H',
}


class QASEnv:
    def __init__(self, max_depth):
        super().__init__()
        self.max_depth = max_depth
        self.reset()

    def reset(self):
        self.actions = []  # List of actions taken (gate indices)
        self.depth = 0  # Current circuit depth
        return self.get_observation()

    def get_observation(self):
        obs = torch.zeros(self.max_depth, dtype=torch.long)
        obs[:len(self.actions)] = torch.tensor(self.actions, dtype=torch.long)
        return obs

    def step(self, action):
        self.actions.append(action)
        self.depth += 1
        done = self.depth >= self.max_depth
        observation = self.get_observation()

        return done, observation

    def get_circuit(self):
        return self.actions


class MuZero(nn.Module):
    def __init__(self, observation_space, action_space_size, hidden_size):
        super().__init__()
        self.observation_space = observation_space
        self.action_space_size = action_space_size

        self.representation_network_1 = nn.Sequential(
            nn.Embedding(action_space_size, hidden_size),
            nn.LSTM(hidden_size, hidden_size, batch_first=True),
        )
        self.representation_network_2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.dynamics_network = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.reward_network = nn.Linear(hidden_size, 1)
        self.policy_network = nn.Linear(hidden_size, action_space_size)
        self.value_network = nn.Linear(hidden_size, 1)

    def representation_network(self, observation):
        output, (hn, cn) = self.representation_network_1(observation)
        x = output[:, -1, :]

        return self.representation_network_2(x)

    def initial_inf(self, observation):
        hidden_state = self.representation_network(observation)
        policy_logits = self.policy_network(hidden_state)
        value = self.value_network(hidden_state)

        return value, policy_logits, hidden_state

    def recurrent_inf(self, hidden_state, action):
        action_onehot = torch.zeros(size=(action.shape[0], 1),
                                    dtype=torch.float32)
        action_onehot[:, 0] = action.float() / self.action_space_size
        x = torch.cat([hidden_state, action_onehot], dim=1)
        next_hidden_state = self.dynamics_network(x)
        reward = self.reward_network(next_hidden_state)
        policy_logits = self.policy_network(next_hidden_state)
        value = self.value_network(next_hidden_state)

        return value, policy_logits, next_hidden_state, reward


class Node:
    def __init__(self, prior):
        self.visit_count = 0
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


def mcts(root, max_depth, muzero, env, discount, num_simulations, x_1, x_2, y):
    for _ in range(num_simulations):
        node = root
        search_path = [node]
        current_env = QASEnv(max_depth)
        current_env.actions = env.get_circuit().copy()

        while node.expanded():
            action, node = select_child(node)
            current_env.step(action)
            search_path.append(node)

        observation = current_env.get_observation().unsqueeze(0)
        value, policy_logits, hidden_state = muzero.initial_inf(observation)
        value = value.item()
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).detach().numpy()

        node.hidden_state = hidden_state
        expand_node(node, action_space_size, policy)

        reward = get_fidelity(current_env.get_circuit(), x_1, x_2, y)
        node.reward = reward

        backpropagate(search_path, value, discount)

    max_visit_count = -1
    best_action = None
    for action, child in root.children.items():
        if child.visit_count > max_visit_count:
            max_visit_count = child.visit_count
            best_action = action

    return best_action


def select_child(node):
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


def ucb(child, parent):
    """pucb(prior upper confidence bound)
    pucb(a) = Q(a) + c * P(a) * sqrt(N)/(1+n),
    a := action
    Q := avg. return q value of a(exploit)
    c := control exploration
    P(a) := prior prob. of a, attained from policy network
    N := visit count of parental node
    n := visit count of current node
    """
    sqrt_N = math.sqrt(parent.visit_count)
    one_plus_n = 1 + child.visit_count
    prior_score = child.prior * sqrt_N / one_plus_n
    value_score = child.value()

    return value_score + prior_score


def expand_node(node, action_space_size, policy):
    for action in range(action_space_size):
        node.children[action] = Node(policy[action])


def backpropagate(search_path, value, discount):
    for node in reversed(search_path):
        node.value_sum += value
        node.visit_count += 1
        value = node.reward + discount * value


def get_fidelity(actions, x_1, x_2, y):
    @qml.qnode(dev)
    def circuit(inputs):
        build_circuit(inputs[0:4], actions)
        qml.adjoint(build_circuit)(inputs[4:8], actions)
        return qml.probs(wires=range(n_qubit))

    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes={})
    x = torch.concat([x_1, x_2], 1)
    pred = qlayer(x)[:, 0]
    fidelity_loss = torch.nn.MSELoss()(pred, y)
    reward = 1 - fidelity_loss.item()
    return reward


def build_circuit(x_data, actions):
    for action in actions:
        gate_info = action_mapping[action]
        apply_gate(gate_info, x_data)


def apply_gate(gate_info, x_data):
    wires = range(n_qubit)
    if isinstance(gate_info, str):
        gate = gate_info
        n = None
    else:
        gate, n = gate_info
    if gate == 'X':
        for qubit in wires:
            qml.PauliX(wires=qubit)
    elif gate == 'Y':
        for qubit in wires:
            qml.PauliY(wires=qubit)
    elif gate == 'Z':
        for qubit in wires:
            qml.PauliZ(wires=qubit)
    elif gate == 'CX':
        for qubit in range(n_qubit - 1):
            qml.CNOT(wires=[qubit, qubit + 1])
    elif gate == 'CY':
        for qubit in range(n_qubit - 1):
            qml.CY(wires=[qubit, qubit + 1])
    elif gate == 'CZ':
        for qubit in range(n_qubit - 1):
            qml.CZ(wires=[qubit, qubit + 1])
    elif gate == 'CRx_pi_over_n':
        angle = np.pi / n
        for qubit in range(n_qubit - 1):
            qml.CRX(angle, wires=[qubit, qubit + 1])
    elif gate == 'CRy_pi_over_n':
        angle = np.pi / n
        for qubit in range(n_qubit - 1):
            qml.CRY(angle, wires=[qubit, qubit + 1])
    elif gate == 'CRz_pi_over_n':
        angle = np.pi / n
        for qubit in range(n_qubit - 1):
            qml.CRZ(angle, wires=[qubit, qubit + 1])
    elif gate == 'Rx_pi_x':
        for qubit in wires:
            qml.RX(np.pi * x_data[qubit], wires=qubit)
    elif gate == 'Ry_pi_x':
        for qubit in wires:
            qml.RY(np.pi * x_data[qubit], wires=qubit)
    elif gate == 'Rz_pi_x':
        for qubit in wires:
            qml.RZ(np.pi * x_data[qubit], wires=qubit)
    elif gate == 'Rx_pi_over_n':
        angle = np.pi / n
        for qubit in wires:
            qml.RX(angle, wires=qubit)
    elif gate == 'Ry_pi_over_n':
        angle = np.pi / n
        for qubit in wires:
            qml.RY(angle, wires=qubit)
    elif gate == 'Rz_pi_over_n':
        angle = np.pi / n
        for qubit in wires:
            qml.RZ(angle, wires=qubit)
    elif gate == 'Rx_arctan_x':
        for qubit in wires:
            qml.RX(np.arctan(x_data[qubit]), wires=qubit)
    elif gate == 'Ry_arctan_x':
        for qubit in wires:
            qml.RY(np.arctan(x_data[qubit]), wires=qubit)
    elif gate == 'Rz_arctan_x':
        for qubit in wires:
            qml.RZ(np.arctan(x_data[qubit]), wires=qubit)
    elif gate == 'H':
        for qubit in wires:
            qml.Hadamard(wires=qubit)



if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    episodes = 10
    batch_size = 25
    max_depth = 8
    num_simulations = 20
    discount = 0.9
    hidden_size = 128


    action_space_size = len(action_mapping)
    X_train, X_test, Y_train, Y_test = dataprep('kmnist', n_qubit)

    env = QASEnv(max_depth=max_depth)
    muzero = MuZero(observation_space=max_depth,
                    action_space_size=action_space_size,
                    hidden_size=hidden_size)
    optimizer = torch.optim.Adam(muzero.parameters(), lr=1e-3)

    for episode in range(episodes):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)

        observation = env.reset().unsqueeze(0)
        done = False
        total_loss = 0

        root_value, policy_logits, hidden_state = muzero.initial_inf(observation)
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).detach().numpy()
        root = Node(0)
        expand_node(root, action_space_size, policy)
        root.hidden_state = hidden_state

        while not done:
            action = mcts(root=root, max_depth=max_depth, muzero=muzero,
                          env=env, discount=discount, num_simulations=25,
                          x_1=X1_batch, x_2=X2_batch, y=Y_batch)
            done, next_observation = env.step(action)
            reward = get_fidelity(env.get_circuit(), X1_batch, X2_batch, Y_batch)

            action_tensor = torch.tensor([action], dtype=torch.float32)
            value, policy_logits, hidden_state = muzero.recurrent_inf(
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
            policy_target = torch.zeros((1, muzero.action_space_size))
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

