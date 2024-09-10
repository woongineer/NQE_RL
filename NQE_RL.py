import gym
import pennylane as qml
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from pennylane import numpy as np
from sklearn.decomposition import PCA


# Load and process data
def data_load_and_process(reduction_size: int = 4):
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

    X_train = PCA(reduction_size).fit_transform(x_train)
    X_test = PCA(reduction_size).fit_transform(x_test)
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
    def __init__(self, state_size, data_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(data_size, data_size ** 2),
            nn.ReLU(),
            nn.Linear(data_size ** 2, data_size ** 2),
            nn.ReLU(),
            nn.Linear(data_size ** 2, data_size)
        )
        self.state_linear_relu_stack = nn.Sequential(
            nn.Linear(state_size, state_size * 2),
            nn.ReLU(),
            nn.Linear(state_size * 2, state_size * 2),
            nn.ReLU(),
            nn.Linear(state_size * 2, state_size)
        )
        self.action_select = nn.Linear(state_size + data_size + data_size,
                                       action_size)

    def forward(self, state, x1, x2):
        x1 = self.linear_relu_stack(x1)
        x2 = self.linear_relu_stack(x2)
        x_state = self.state_linear_relu_stack(state)
        x = torch.concat([x1, x2, x_state], 1)
        action_probs = torch.softmax(self.action_select(x), dim=-1)

        return action_probs


class QASEnv(gym.Env):
    def __init__(
            self,
            num_of_qubit: int = 4,
            fidelity_threshold: float = 0.95,
            reward_penalty: float = 0.01,
            max_timesteps: int = 20,
    ):
        super().__init__()
        self.simulator = qml.device('default.qubit', wires=num_of_qubit)
        self.qubits = self.simulator.wires.tolist()
        self.fidelity_threshold = fidelity_threshold
        self.reward_penalty = reward_penalty
        self.max_timesteps = max_timesteps
        self.action_gate = []

    def reset(self):
        self.circuit_gates = []
        return self.get_obs()

    def select_action(self, action, input):  # 여기다가 input adj??
        action_gates = []
        for idx, qubit in enumerate(self.qubits):
            next_qubit = self.qubits[(idx + 1) % len(self.qubits)]
            if action == 0:
                action_gates += [qml.Hadamard(wires=idx)]
            elif action == 1:
                for j in range(input.shape[-1]):
                    action_gates += [qml.RX(input[j], wires=idx)]
            elif action == 2:
                for j in range(input.shape[-1]):
                    action_gates += [qml.RY(input[j], wires=idx)]
            elif action == 3:
                for j in range(input.shape[-1]):
                    action_gates += [qml.RZ(input[j], wires=idx)]
            elif action == 4:
                action_gates += [qml.CNOT(wires=[qubit, next_qubit])]
        self.action_gate = action_gates

    def get_obs(self):

        dev = qml.device("default.qubit", wires=self.qubits)

        gates = self.action_gate

        @qml.qnode(dev)
        def circuit(pauli):
            for qubit in range(len(self.qubits)):
                qml.Identity(wires=qubit)
            # qml.RX(np.pi / 7, wires=3)  ##TODO Remove
            # qml.RY(np.pi / 7, wires=1)
            # qml.RZ(np.pi / 7, wires=0)

            for gate in gates:
                gate

            if pauli == 'X':
                return [qml.expval(qml.PauliX(wires=w)) for w in
                        range(len(self.qubits))]

            elif pauli == 'Y':
                return [qml.expval(qml.PauliY(wires=w)) for w in
                        range(len(self.qubits))]

            elif pauli == 'Z':
                return [qml.expval(qml.PauliZ(wires=w)) for w in
                        range(len(self.qubits))]

        return np.concatenate((circuit('X'), circuit('Y'), circuit('Z')))

    # def get_fidelity(self):
    #   뭔가 get_pennylane이 처음부터 만드는 것처럼 adj도 처음부터 만들어도 될 거 같은데....
    #     dev = qml.device("default.qubit", wires=self.qubits)
    #
    #     @qml.qnode(dev)
    #     def circuit():
    #         for gate in gates:
    #             gate
    #         return qml.state()
    #
    #     return circuit

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
    data_size = 4  # JW Data reduction size from 256->, determine # of qubit
    gamma = 0.98
    learning_rate = 0.01
    state_size = 3 * data_size  # *3 because of Pauli X,Y,Z
    action_size = 5  # Number of possible actions, RX, RY, RZ, H, CX
    episodes = 10
    iterations = 7
    batch_size = 25

    # Load data
    X_train, X_test, Y_train, Y_test = data_load_and_process(reduction_size=data_size)
    policy = PolicyNetwork(state_size=state_size,
                           data_size=data_size,
                           action_size=action_size)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    env = QASEnv(num_of_qubit=data_size)

    for episode in range(episodes):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        state = env.reset()
        done = False
        log_probs = []
        rewards = []

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = policy.forward(state_tensor, X1_batch, X2_batch)
            dist = torch.distributions.Categorical(probs[0])
            action = dist.sample().item()

            next_state, reward, done, info = env.step(action)

            log_probs.append(dist.log_prob(torch.tensor(action)))
            rewards.append(reward)

            state = next_state

        # Compute return
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # Convert returns to tensor
        returns = torch.tensor(returns, dtype=torch.float32)

        # Update policy network
        policy_loss = -torch.stack(log_probs).float() * returns
        policy_loss = policy_loss.mean()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        print(f'Episode {episode + 1}/{episodes} complete.')

    print('Training Complete')
