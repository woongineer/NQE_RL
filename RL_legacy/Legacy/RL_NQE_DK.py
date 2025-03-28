import gym
import matplotlib.pyplot as plt
import pennylane as qml
import seaborn as sns
import tensorflow as tf
import torch
import torch.optim as optim
from pennylane import numpy as np
from sklearn.decomposition import PCA
from torch import nn


# Load and process data
def data_load_and_process(dataset, reduction_size: int = 4):
    if dataset == 'mnist':
        (x_train, y_train), (
            x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'kmnist':
        # Path to training images and corresponding labels provided as numpy arrays
        kmnist_train_images_path = "/RL_legacy/kmnist/kmnist-train-imgs.npz"
        kmnist_train_labels_path = "/RL_legacy/kmnist/kmnist-train-labels.npz"

        # Path to the test images and corresponding labels
        kmnist_test_images_path = "/RL_legacy/kmnist/kmnist-test-imgs.npz"
        kmnist_test_labels_path = "/RL_legacy/kmnist/kmnist-test-labels.npz"

        x_train = np.load(kmnist_train_images_path)['arr_0']
        y_train = np.load(kmnist_train_labels_path)['arr_0']

        # Load the test data from the corresponding npz files
        x_test = np.load(kmnist_test_images_path)['arr_0']
        y_test = np.load(kmnist_test_labels_path)['arr_0']

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

    # X1_new 처리
    X1_new_array = np.array(X1_new)
    X1_new_tensor = torch.from_numpy(X1_new_array).float()

    # X2_new 처리
    X2_new_array = np.array(X2_new)
    X2_new_tensor = torch.from_numpy(X2_new_array).float()

    # Y_new 처리
    Y_new_array = np.array(Y_new)
    Y_new_tensor = torch.from_numpy(Y_new_array).float()
    return X1_new_tensor, X2_new_tensor, Y_new_tensor


def state_to_tensor(state):
    pennylane_to_torch = [torch.tensor(i) for i in state]
    stacked = torch.stack(pennylane_to_torch)
    return stacked.float()


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.state_linear_relu_stack = nn.Sequential(
            nn.Linear(state_size, state_size * 2),
            nn.ReLU(),
            nn.Linear(state_size * 2, state_size * 2),
            nn.ReLU(),
            nn.Linear(state_size * 2, state_size)
        )
        self.action_select = nn.Linear(state_size, action_size)

    def forward(self, state):
        state_new = self.state_linear_relu_stack(state)
        action_probs = torch.softmax(self.action_select(state_new), dim=-1)

        epsilon = 0.05
        adjust_action_probs = (action_probs + epsilon) / (
                    1 + epsilon * action_size)

        return adjust_action_probs


class QASEnv(gym.Env):
    def __init__(
            self,
            num_of_qubit: int = 4,
            max_timesteps: int = 14 * 3,  # N_layers == 3
            batch_size: int = 25
    ):
        super().__init__()
        self.simulator = qml.device('default.qubit', wires=num_of_qubit)
        self.qubits = self.simulator.wires.tolist()
        self.max_timesteps = max_timesteps
        self.batch_size = batch_size

    def reset(self):
        self.circuit_gates_x1 = [
            self.select_action([None for _ in range(self.batch_size)], None)]
        self.circuit_gates_x2 = [
            self.select_action([None for _ in range(self.batch_size)], None)]
        return self.get_obs()

    def select_action(self, action, input):
        action_set = []
        for i in range(self.batch_size):
            action_per_batch = []
            for idx, qubit in enumerate(self.qubits):
                next_qubit = self.qubits[(idx + 1) % len(self.qubits)]
                if action[i] == None:
                    action_per_batch += [qml.Identity(wires=idx)]
                elif action[i] == 0:
                    action_per_batch += [qml.Hadamard(wires=idx)]
                elif action[i] == 1:
                    action_per_batch += [
                        qml.RX(input[i][idx].resolve_conj().numpy(), wires=idx)]
                elif action[i] == 2:
                    action_per_batch += [
                        qml.RY(input[i][idx].resolve_conj().numpy(), wires=idx)]
                elif action[i] == 3:
                    action_per_batch += [
                        qml.RZ(input[i][idx].resolve_conj().numpy(), wires=idx)]
                elif action[i] == 4:
                    action_per_batch += [qml.CNOT(wires=[qubit, next_qubit])]
            action_set += [action_per_batch]
        return action_set

    def get_obs(self):

        dev = qml.device("default.qubit", wires=self.qubits)

        gates_x1 = [list(row) for row in zip(*self.circuit_gates_x1)]
        gates_x2 = [list(row) for row in zip(*self.circuit_gates_x2)]

        @qml.qnode(dev)
        def circuit(batch_x1, batch_x2):
            for seq in batch_x1:
                for gate in seq:
                    gate.queue()
            for seq in batch_x2[::-1]:
                for gate in seq:
                    qml.adjoint(gate).queue()

            return qml.probs(wires=range(len(self.qubits)))

        measure_probs = []
        measure_0s = []

        for batch_x1, batch_x2 in zip(gates_x1, gates_x2):
            measure_prob = circuit(batch_x1, batch_x2)
            measure_probs.append(measure_prob)
            measure_0s.append(measure_prob[0])

        return measure_probs, measure_0s

    def step(self, action, X1, X2, Y_batch):
        action_gate_x1 = self.select_action(action, X1)
        action_gate_x2 = self.select_action(action, X2)

        self.circuit_gates_x1.append(action_gate_x1)
        self.circuit_gates_x2.append(action_gate_x2)

        measure_probs, measure_0s = self.get_obs()

        loss_fn = torch.nn.MSELoss(reduction='none')
        measure_0s = torch.stack([torch.tensor(i) for i in measure_0s])
        measure_loss = loss_fn(measure_0s, Y_batch)

        reward = -measure_loss
        terminal = len(self.circuit_gates_x1) >= self.max_timesteps

        return measure_probs, reward, terminal


dev = qml.device('default.qubit', wires=4)


def exp_Z(x, wires):
    qml.RZ(-2 * x, wires=wires)


# exp(i(pi - x1)(pi - x2)ZZ) gate
def exp_ZZ2(x1, x2, wires):
    qml.CNOT(wires=wires)
    qml.RZ(-2 * (np.pi - x1) * (np.pi - x2), wires=wires[1])
    qml.CNOT(wires=wires)


# Quantum Embedding 1 for model 1 (Conventional ZZ feature embedding)
def QuantumEmbedding_ZZ(input):
    for i in range(N_layers):
        for j in range(4):
            qml.Hadamard(wires=j)
            exp_Z(input[j], wires=j)
        for k in range(3):
            exp_ZZ2(input[k], input[k + 1], wires=[k, k + 1])
        exp_ZZ2(input[3], input[0], wires=[3, 0])


@qml.qnode(dev, interface="torch")
def circuit_ZZ(inputs):
    QuantumEmbedding_ZZ(inputs[0:4])
    qml.adjoint(QuantumEmbedding_ZZ)(inputs[4:8])
    return qml.probs(wires=range(4))


class x_transform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

    def forward(self, x):
        x = self.linear_relu_stack1(x)
        return x.detach().numpy()


def statepreparation(x, NQE, embedding_type):
    if NQE:
        if embedding_type == 'RL_legacy':
            x = model_transform_RL(torch.tensor(x))
        elif embedding_type == 'ZZ':
            x = model_transform_ZZ(torch.tensor(x))
    if embedding_type == 'RL_legacy':
        QuantumEmbedding_RL(action_list, x)
    elif embedding_type == 'ZZ':
        QuantumEmbedding_ZZ(x)


def U_SU4(params, wires):  # 15 params
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])


def QCNN(params):
    param1 = params[0:15]
    param2 = params[15:30]

    U_SU4(param1, wires=[0, 1])
    U_SU4(param1, wires=[2, 3])
    U_SU4(param1, wires=[1, 2])
    U_SU4(param1, wires=[3, 0])
    U_SU4(param2, wires=[0, 2])


@qml.qnode(dev)
def QCNN_classifier(params, x, NQE, embedding_type):
    statepreparation(x, NQE, embedding_type)
    QCNN(params)
    return qml.expval(qml.PauliZ(2))


def Linear_Loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss += 0.5 * (1 - l * p)
    return loss / len(labels)


def cost(weights, X_batch, Y_batch, Trained, embedding_type):
    preds = [QCNN_classifier(weights, x, Trained, embedding_type) for x in
             X_batch]
    return Linear_Loss(Y_batch, preds)


def circuit_training(X_train, Y_train, Trained, embedding_type):
    weights = np.random.random(30, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
    loss_history = []
    for it in range(steps):
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]
        weights, cost_new = opt.step_and_cost(
            lambda v: cost(v, X_batch, Y_batch, Trained, embedding_type),
            weights)
        loss_history.append(cost_new)
        if it % 3 == 0:
            print(f"iteration:{it} cost:{cost_new}, "
                  f"embedding_type:{embedding_type}")
    return loss_history, weights


def accuracy_test(predictions, labels):
    acc = 0
    for l, p in zip(labels, predictions):
        if np.abs(l - p) < 1:
            acc = acc + 1
    return acc / len(labels)


def QuantumEmbedding_RL(action_list, input):
    for action in action_list:
        for j in range(4):
            if action == None:
                qml.Identity(wires=j)
            elif action == 0:
                qml.Hadamard(wires=j)
            elif action == 1:
                qml.RX(input[j], wires=j)
            elif action == 2:
                qml.RY(input[j], wires=j)
            elif action == 3:
                qml.RZ(input[j], wires=j)
            elif action == 4:
                qml.CNOT(wires=[j, (j + 1) % 4])


class Model_Fidelity_ZZ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer1 = qml.qnn.TorchLayer(circuit_ZZ, weight_shapes={})
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
        """you can use 
        fig, ax = qml.draw_mpl(circuit)(x)
        fig.savefig('RL_legacy/dd.png')
        to see the circuit
        """
        return x[:, 0]


class Model_Fidelity_RL(torch.nn.Module):
    def __init__(self, action_list):
        super().__init__()
        self.qlayer1 = qml.qnn.TorchLayer(self.circuitRL, weight_shapes={})
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.action_list = action_list

    def forward(self, x1, x2):
        x1 = self.linear_relu_stack1(x1)
        x2 = self.linear_relu_stack1(x2)
        x = torch.concat([x1, x2], 1)
        x = self.qlayer1(x)
        """you can use 
        fig, ax = qml.draw_mpl(circuit)(x)
        fig.savefig('RL_legacy/dd.png')
        to see the circuit
        """
        return x[:, 0]

    @qml.qnode(dev, interface="torch")
    def circuitRL(inputs):
        QuantumEmbedding_RL(action_list, inputs[0:4])
        qml.adjoint(QuantumEmbedding_RL)(action_list, inputs[4:8])
        return qml.probs(wires=range(4))


if __name__ == "__main__":
    # Hyperparameters
    data_size = 4  # JW Data reduction size from 256->, determine # of qubit
    gamma = 0.98
    learning_rate = 0.01
    state_size = data_size ** 2
    action_size = 5  # Number of possible actions, RX, RY, RZ, H, CX
    episodes = 3
    iterations = 7
    steps = 7
    batch_size = 7
    N_layers = 3

    # Load data
    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='mnist',
                                                             reduction_size=data_size)
    policy = PolicyNetwork(state_size=state_size, action_size=action_size)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    env = QASEnv(num_of_qubit=data_size, max_timesteps=14 * N_layers,
                 batch_size=batch_size)

    policy_losses = []

    for episode in range(episodes):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        state, _ = env.reset()
        done = False
        log_probs = []
        rewards = []
        action_list = []
        prev_action = torch.tensor([999 for _ in range(batch_size)])

        while not done:
            state_tensor = state_to_tensor(state)
            prob = policy.forward(state_tensor)
            dist = torch.distributions.Categorical(prob)

            action = dist.sample()
            mask = (action == prev_action)
            trial = 0
            while mask.any():
                trial += 1
                if trial > 0:
                    print(f"{trial}th trial, {action}")
                new_samples = dist.sample()
                action[mask] = new_samples[mask]
                mask = (action == prev_action)
            prev_action = action

            next_state, reward, done = env.step(action, X1_batch, X2_batch,
                                                Y_batch)

            log_prob = dist.log_prob(torch.tensor(action))
            log_probs.append(log_prob)
            rewards.append(reward)
            action_list.append(action)

            state = next_state

        rewards = torch.stack(rewards)  # Shape: [num_steps, batch_size]
        returns = torch.zeros_like(rewards)
        G = torch.zeros(rewards.size(1))  # Shape: [batch_size]
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            returns[t] = G
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy loss
        log_probs = torch.stack(log_probs)  # Shape: [num_steps, batch_size]
        policy_loss = -log_probs * returns
        policy_loss = policy_loss.mean()
        policy_losses.append(policy_loss)

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        print(
            f'E{episode + 1}/{episodes}, loss:{policy_loss}, actions:{action_list}')

        early_stop = 7
        if len(policy_losses) >= early_stop:
            last_losses = [loss.detach().item() for loss in
                           policy_losses[-early_stop:]]
            if len(set(last_losses)) == 1:
                print('Episode early stopped')
                break

    print('Training Complete')

    policy_losses = [loss.detach().numpy() for loss in policy_losses]
    plt.plot(policy_losses)
    plt.savefig('/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/RL_legacy/policy_loss.png')

    model_RL = Model_Fidelity_RL(action_list)
    model_ZZ = Model_Fidelity_ZZ()
    model_RL.train()
    model_ZZ.train()

    loss_fn = torch.nn.MSELoss()
    opt_RL = torch.optim.SGD(model_RL.parameters(), lr=0.01)
    opt_ZZ = torch.optim.SGD(model_ZZ.parameters(), lr=0.01)
    for it in range(iterations):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        pred_RL = model_RL(X1_batch, X2_batch)
        pred_ZZ = model_ZZ(X1_batch, X2_batch)
        loss_RL = loss_fn(pred_RL, Y_batch)
        loss_ZZ = loss_fn(pred_ZZ, Y_batch)

        opt_RL.zero_grad()
        opt_ZZ.zero_grad()
        loss_RL.backward()
        loss_ZZ.backward()
        opt_RL.step()
        opt_ZZ.step()

        if it % 3 == 0:
            print(
                f"Iterations: {it} Loss_RL: {loss_RL.item()} Loss_ZZ: {loss_ZZ.item()}")

    torch.save(model_RL.state_dict(), "/RL_legacy/model_RL.pt")
    torch.save(model_ZZ.state_dict(), "/RL_legacy/model_ZZ.pt")

    Y_train = [-1 if y == 0 else 1 for y in Y_train]
    Y_test = [-1 if y == 0 else 1 for y in Y_test]

    model_transform_RL = x_transform()
    model_transform_ZZ = x_transform()
    model_transform_RL.load_state_dict(
        torch.load("/RL_legacy/model_RL.pt", weights_only=True))
    model_transform_ZZ.load_state_dict(
        torch.load("/RL_legacy/model_ZZ.pt", weights_only=True))

    loss_history_without_NQE_RL, weight_without_NQE_RL = circuit_training(
        X_train,
        Y_train,
        Trained=False,
        embedding_type='RL_legacy')
    loss_history_with_NQE_RL, weight_with_NQE_RL = circuit_training(X_train,
                                                                    Y_train,
                                                                    Trained=True,
                                                                    embedding_type='RL_legacy')
    loss_history_without_NQE_ZZ, weight_without_NQE_ZZ = circuit_training(
        X_train,
        Y_train,
        Trained=False,
        embedding_type='ZZ')
    loss_history_with_NQE_ZZ, weight_with_NQE_ZZ = circuit_training(X_train,
                                                                    Y_train,
                                                                    Trained=True,
                                                                    embedding_type='ZZ')

    plt.rcParams['figure.figsize'] = [10, 5]
    fig, ax = plt.subplots()
    clrs = sns.color_palette("husl", 4)
    with sns.axes_style("darkgrid"):
        ax.plot(range(len(loss_history_without_NQE_RL)),
                loss_history_without_NQE_RL,
                label="Without NQE_RL", c=clrs[0])
        ax.plot(range(len(loss_history_with_NQE_RL)), loss_history_with_NQE_RL,
                label="With NQE_RL", c=clrs[1])
        ax.plot(range(len(loss_history_without_NQE_ZZ)),
                loss_history_without_NQE_ZZ,
                label="Without NQE_ZZ", c=clrs[2])
        ax.plot(range(len(loss_history_with_NQE_ZZ)), loss_history_with_NQE_ZZ,
                label="With NQE_ZZ", c=clrs[3])

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("QCNN Loss Histories")
    ax.legend()

    fig.savefig('/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/RL_legacy/QCNN_loss_history_compare.png')

    accuracies_without_NQE_RL, accuracies_with_NQE_RL = [], []
    accuracies_without_NQE_ZZ, accuracies_with_NQE_ZZ = [], []

    prediction_without_NQE_RL = [
        QCNN_classifier(weight_without_NQE_RL, x, NQE=False,
                        embedding_type='RL_legacy')
        for x in X_test]
    prediction_with_NQE_RL = [
        QCNN_classifier(weight_with_NQE_RL, x, NQE=True, embedding_type='RL_legacy')
        for x
        in X_test]
    prediction_without_NQE_ZZ = [
        QCNN_classifier(weight_without_NQE_ZZ, x, NQE=False,
                        embedding_type='ZZ')
        for x in X_test]
    prediction_with_NQE_ZZ = [
        QCNN_classifier(weight_with_NQE_ZZ, x, NQE=True, embedding_type='ZZ')
        for x
        in X_test]

    accuracy_without_NQE_RL = accuracy_test(prediction_without_NQE_RL,
                                            Y_test) * 100
    accuracy_with_NQE_RL = accuracy_test(prediction_with_NQE_RL, Y_test) * 100
    accuracy_without_NQE_ZZ = accuracy_test(prediction_without_NQE_ZZ,
                                            Y_test) * 100
    accuracy_with_NQE_ZZ = accuracy_test(prediction_with_NQE_ZZ, Y_test) * 100

    print(f"Accuracy without NQE_RL: {accuracy_without_NQE_RL:.3f}")
    print(f"Accuracy with NQE_RL: {accuracy_with_NQE_RL:.3f}")
    print(f"Accuracy without NQE_ZZ: {accuracy_without_NQE_ZZ:.3f}")
    print(f"Accuracy with NQE_ZZ: {accuracy_with_NQE_ZZ:.3f}")

    print('end')
