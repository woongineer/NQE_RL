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


def state_to_tensor(state):
    pennylane_to_torch = [torch.tensor(i) for i in state]
    stacked = torch.stack(pennylane_to_torch)
    return torch.tensor(stacked, dtype=torch.float32)


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

    def reset(self):
        self.circuit_gates_x1 = [
            self.select_action([0 for _ in range(25)], None)]
        self.circuit_gates_x2 = [
            self.select_action([0 for _ in range(25)], None)]
        return self.get_obs()

    def select_action(self, action, input):  # 여기다가 input adj??
        action_set = []
        for i in range(len(action)):  # JW batch size 만큼 iter
            action_per_batch = []
            for idx, qubit in enumerate(self.qubits):
                next_qubit = self.qubits[(idx + 1) % len(self.qubits)]
                if action[i] == 0:
                    action_per_batch += [qml.Identity(wires=idx)]
                elif action[i] == 1:
                    action_per_batch += [qml.Hadamard(wires=idx)]
                elif action[i] == 2:
                    action_per_batch += [
                        qml.RX(input[i][idx].resolve_conj().numpy(), wires=idx)]
                elif action[i] == 3:
                    action_per_batch += [
                        qml.RY(input[i][idx].resolve_conj().numpy(), wires=idx)]
                elif action[i] == 4:
                    action_per_batch += [
                        qml.RZ(input[i][idx].resolve_conj().numpy(), wires=idx)]
                elif action[i] == 5:
                    action_per_batch += [qml.CNOT(wires=[qubit, next_qubit])]
            action_set += [action_per_batch]
        return action_set

    def get_obs(self):

        dev = qml.device("default.qubit", wires=self.qubits)

        gates_x1 = [list(row) for row in zip(*self.circuit_gates_x1)]
        gates_x2 = [list(row) for row in zip(*self.circuit_gates_x2)]

        @qml.qnode(dev)
        def circuit(pauli, batch_x1, batch_x2):
            for seq in batch_x1:
                for gate in seq:
                    gate.queue()
            for seq in batch_x2[::-1]:
                for gate in seq:
                    qml.adjoint(gate).queue()

            if pauli == 'X':
                return [qml.expval(qml.PauliX(wires=w)) for w in
                        range(len(self.qubits))]
                # return [qml.expval(qml.PauliX(wires=range(len(self.qubits))))]

            elif pauli == 'Y':
                return [qml.expval(qml.PauliY(wires=w)) for w in
                        range(len(self.qubits))]

            elif pauli == 'Z':
                return [qml.expval(qml.PauliZ(wires=w)) for w in
                        range(len(self.qubits))]

            elif pauli == 'F':
                return qml.probs(wires=range(len(self.qubits)))

            ## TODO 여기서 바로 prob 줄까?

        pauli_measure = []
        fidelity = []
        for batch_x1, batch_x2 in zip(gates_x1, gates_x2):
            x_obs = circuit('X', batch_x1, batch_x2)
            y_obs = circuit('Y', batch_x1, batch_x2)
            z_obs = circuit('Z', batch_x1, batch_x2)
            pauli_measure.append(np.concatenate((x_obs, y_obs, z_obs)))

            fidelity.append(circuit('F', batch_x1, batch_x2)[0])

        return pauli_measure, fidelity

    def step(self, action, X1, X2, Y_batch):
        action_gate_x1 = self.select_action(action, X1)
        action_gate_x2 = self.select_action(action, X2)

        self.circuit_gates_x1.append(action_gate_x1)
        self.circuit_gates_x2.append(action_gate_x2)

        observation, fidelity = self.get_obs()

        fidelity = torch.stack([torch.tensor(i) for i in fidelity])
        fidelity = torch.nn.MSELoss()(fidelity, Y_batch)

        reward = fidelity - self.reward_penalty if fidelity > self.fidelity_threshold else -self.reward_penalty
        terminal = (reward > 0.) or (
                len(self.circuit_gates_x1) >= self.max_timesteps)

        return observation, reward, terminal


if __name__ == "__main__":
    # Hyperparameters
    data_size = 4  # JW Data reduction size from 256->, determine # of qubit
    gamma = 0.98
    learning_rate = 0.01
    state_size = 3 * data_size  # *3 because of Pauli X,Y,Z
    action_size = 6  # Number of possible actions, RX, RY, RZ, H, CX
    episodes = 10
    iterations = 7
    batch_size = 25

    # Load data
    X_train, X_test, Y_train, Y_test = data_load_and_process(
        reduction_size=data_size)
    policy = PolicyNetwork(state_size=state_size,
                           data_size=data_size,
                           action_size=action_size)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    env = QASEnv(num_of_qubit=data_size)

    for episode in range(episodes):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        state, _ = env.reset()
        done = False
        log_probs = []
        rewards = []

        while not done:
            state_tensor = state_to_tensor(state)
            probs = policy.forward(state_tensor, X1_batch, X2_batch)
            dists = [torch.distributions.Categorical(prob) for prob in probs]
            actions = [dist.sample().item() for dist in dists]

            next_state, reward, done = env.step(actions, X1_batch,
                                                X2_batch, Y_batch)


            log_probs.append([d.log_prob(torch.tensor(a)) for a, d in zip(actions, dists)])
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
