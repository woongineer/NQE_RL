import gym
import pennylane as qml
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from pennylane import numpy as np
from sklearn.decomposition import PCA

# Define the quantum device
# dev = qml.device('default.qubit', wires=4)


# Load and process data
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


# Make new data for hybrid model
def new_data(batch_size, X, Y):
    X1_new, X2_new, Y_new = [], [], []
    for i in range(batch_size):
        n, m = np.random.randint(len(X)), np.random.randint(len(X))
        X1_new.append(X[n])
        X2_new.append(X[m])
        Y_new.append(1 if Y[n] == Y[m] else 0)
    return torch.tensor(X1_new).float(), torch.tensor(
        X2_new).float(), torch.tensor(Y_new).float()


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs


class QASEnv(gym.Env):
    def __init__(
            self,
            qubits: list[str] = None,  ##TODO Check Type
            action_gates: list = None,  ##TODO Check Type
            fidelity_threshold: float = 0.95,
            reward_penalty: float = 0.01,
            max_timesteps: int = 20,
    ):
        super().__init__()
        self.simulator = qml.device('default.qubit', wires=4)
        if qubits is None:
            qubits = self.simulator.wires.tolist()
        self.qubits = qubits
        if action_gates is None:
            action_gates = []
            for idx, qubit in enumerate(qubits):
                next_qubit = qubits[(idx + 1) % len(qubits)]
                ##TODO Continuous를 어떻게 넣지??
                action_gates += [
                    qml.Hadamard(wires=qubit),
                    qml.RX(np.pi, wires=qubit),
                    qml.RX(np.pi / 2, wires=qubit),
                    qml.RX(np.pi / 3, wires=qubit),
                    qml.RX(np.pi / 8, wires=qubit),
                    qml.RY(np.pi, wires=qubit),
                    qml.RY(np.pi / 2, wires=qubit),
                    qml.RY(np.pi / 3, wires=qubit),
                    qml.RY(np.pi / 8, wires=qubit),
                    qml.RZ(np.pi, wires=qubit),
                    qml.RZ(np.pi / 2, wires=qubit),
                    qml.RZ(np.pi / 3, wires=qubit),
                    qml.RZ(np.pi / 8, wires=qubit),
                    qml.CNOT(wires=[qubit, next_qubit]),
                ]
        self.action_gates = action_gates
        self.fidelity_threshold = fidelity_threshold
        self.reward_penalty = reward_penalty
        self.max_timesteps = max_timesteps

    def reset(self):
        self.circuit_gates = []
        return self.get_obs()

    def get_obs(self):
        circuit = self.get_pennylane()
        # observable Pauli XYZ로 circuit을 측정해야 한다....
        # get_pennylane이 현재 circuit을 가져와야 하는데 그걸 어떻게 하지??
        # obs = [qml.expval(circuit) for obs in observables]
        observables = []
        for qubit in range(self.num_qubits):
            observables.append(qml.PauliX(wires=qubit))
            observables.append(qml.PauliY(wires=qubit))
            observables.append(qml.PauliZ(wires=qubit))

        exp_vals = [circuit(obs) for obs in observables]
        return np.array(exp_vals).real

    """get_pennylane이 QuantumEmbedding같은 역할을 해야 하고
    get_fidelity가 circuit같은 역할을 해야 한다...
    """
    # def QuantumEmbedding(inputs, action):
    #     if action == 0:  # RX applied to all qubits
    #         for i in range(4):
    #             qml.RX(inputs[i], wires=i)
    #     elif action == 1:  # RY applied to all qubits
    #         for i in range(4):
    #             qml.RY(inputs[i], wires=i)
    #     elif action == 2:  # RZ applied to all qubits
    #         for i in range(4):
    #             qml.RZ(inputs[i], wires=i)
    #     elif action == 3:  # CNOT in linear nearest-neighbor configuration
    #         for i in range(3):  # CNOT from qubit i to i+1
    #             qml.CNOT(wires=[i, i + 1])
    #
    # @qml.qnode(dev, interface="torch")
    # def circuit(action, inputs):
    #     self.QuantumEmbedding(inputs[0:4], action)
    #     qml.adjoint(QuantumEmbedding)(inputs[4:8], action)
    #     return qml.probs(wires=range(4))

    def step(self, action):
        action_gate = self.action_gates[action]
        self.circuit_gates.append(action_gate)
        observation = self.get_obs()
        fidelity = self.get_fidelity()
        reward = fidelity - self.reward_penalty if fidelity > self.fidelity_threshold else -self.reward_penalty
        terminal = (reward > 0.) or (
                len(self.circuit_gates) >= self.max_timesteps)
        info = {'fidelity': fidelity, 'circuit': self.get_pennylane()}

        return observation, reward, terminal, info




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

    env = QASEnv()

    for episode in range(episodes):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        state = env.reset()
        done = False
        log_probs = []
        rewards = []

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = policy(state_tensor)
            dist = torch.distributions.Categorical(probs[0])
            action = dist.sample().item()

            next_state, reward, done, info = env.step(action)

            log_probs.append(dist.log_prob(torch.tensor(action)))
            rewards.append(reward)

            state = next_state
