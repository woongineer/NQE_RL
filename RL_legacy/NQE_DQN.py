import random
from collections import deque, namedtuple

import gym
import matplotlib.pyplot as plt
import pennylane as qml
import tensorflow as tf
import torch
from pennylane import numpy as np
from sklearn.decomposition import PCA
from torch import nn

# Set your device
dev = qml.device('default.qubit', wires=4)

def data_load_and_process(dataset='mnist', reduction_size: int = 4):
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


def exp_Z(x, wires):
    qml.RZ(-2 * x, wires=wires)


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

# Define the NQE Model
class NQEModel(torch.nn.Module):
    def __init__(self, action_sequence=None):
        super().__init__()
        self.action_sequence = action_sequence
        if action_sequence is None:
            # Use quantum_embedding_zz
            @qml.qnode(dev, interface="torch")
            def circuit(inputs):
                quantum_embedding_zz(inputs[0:4])
                qml.adjoint(quantum_embedding_zz)(inputs[4:8])
                return qml.probs(wires=range(4))
        else:
            # Use quantum_embedding_rl
            @qml.qnode(dev, interface="torch")
            def circuit(inputs):
                quantum_embedding_rl(inputs[0:4], self.action_sequence)
                qml.adjoint(quantum_embedding_rl)(inputs[4:8], self.action_sequence)
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
def train_NQE(X_train, Y_train, NQE_iterations, batch_size, action_sequence=None):
    NQE_model = NQEModel(action_sequence)
    NQE_model.train()
    NQE_loss_fn = torch.nn.MSELoss()
    NQE_opt = torch.optim.SGD(NQE_model.parameters(), lr=0.01)
    NQE_losses = []
    for it in range(NQE_iterations):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        pred = NQE_model(X1_batch, X2_batch)
        loss = NQE_loss_fn(pred, Y_batch)

        NQE_opt.zero_grad()
        loss.backward()
        NQE_opt.step()

        if it % 3 == 0:
            print(f"Iterations: {it} Loss: {loss.item()}")
        NQE_losses.append(loss.item())
    return NQE_model, NQE_losses

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

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, num_of_qubits):
        super(DQNNetwork, self).__init__()
        self.state_linear_relu_stack = nn.Sequential(
            nn.Linear(state_size * 4, state_size * 8),
            nn.ReLU(),
            nn.Linear(state_size * 8, state_size * 4),
        )
        # 각 큐빗에 대한 Q-값을 출력하는 레이어
        self.action_value_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(state_size * 4, action_size * 2),
                nn.ReLU(),
                nn.Linear(action_size * 2, action_size),
            ) for _ in range(num_of_qubits)]
        )

    def forward(self, state):
        # state: [batch_size, state_size * 4]
        state_new = self.state_linear_relu_stack(state)
        q_values = []
        for qubit_action_value_layer in self.action_value_layers:
            q_value = qubit_action_value_layer(state_new)  # [batch_size, action_size]
            q_values.append(q_value)
        # q_values를 [batch_size, num_of_qubits, action_size]로 변환
        q_values = torch.stack(q_values, dim=0)  # [batch_size, num_of_qubits, action_size]
        return q_values  # [batch_size, num_of_qubits, action_size]



class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        # Transition의 요소별로 묶어서 반환
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)


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
        dummy_action = [None for _ in range(len(self.qubits))]
        dummy_input = [[0 for _ in self.qubits] for _ in range(batch_size)]

        self.circuit_gates_x1 = [self.select_action(dummy_action, dummy_input)]
        self.circuit_gates_x2 = [self.select_action(dummy_action, dummy_input)]
        self.circuit_gates_x = []
        return self.get_obs()

    def select_action(self, action, input):
        action_sets = []

        for input_batch in input:
            action_set = []
            for qubit in self.qubits:
                next_qubit = (qubit + 1) % len(self.qubits)
                if action[qubit] is None:
                    action_set += [qml.Identity(wires=qubit)]
                elif action[qubit] == 0:
                    action_set += [qml.Hadamard(wires=qubit)]
                elif action[qubit] == 1:
                    action_set += [qml.RX(input_batch[qubit], wires=qubit)]
                elif action[qubit] == 2:
                    action_set += [qml.RY(input_batch[qubit], wires=qubit)]
                elif action[qubit] == 3:
                    action_set += [qml.RZ(input_batch[qubit], wires=qubit)]
                elif action[qubit] == 4:
                    action_set += [qml.CNOT(wires=[qubit, next_qubit])]
            action_sets.append(action_set)
        return action_sets

    def compute_state_stats(self, measure_probs):
        measure_probs_tensor = torch.stack(
            [torch.tensor(mp) for mp in measure_probs])

        mean_measure_probs = torch.mean(measure_probs_tensor, dim=0)
        var_measure_probs = torch.var(measure_probs_tensor, dim=0)
        skew_measure_probs = torch.mean(
            ((measure_probs_tensor - mean_measure_probs) ** 3), dim=0) / (
                                     var_measure_probs ** 1.5 + 1e-8)
        kurt_measure_probs = torch.mean(
            ((measure_probs_tensor - mean_measure_probs) ** 4), dim=0) / (
                                     var_measure_probs ** 2 + 1e-8) - 3
        state_stats = torch.cat((mean_measure_probs, var_measure_probs,
                                 skew_measure_probs, kurt_measure_probs), dim=0)

        return state_stats.float()

    def get_obs(self):

        dev = qml.device("default.qubit", wires=self.qubits)

        gates_x1 = [list(row) for row in zip(*self.circuit_gates_x1)]
        gates_x2 = [list(row) for row in zip(*self.circuit_gates_x2)]

        @qml.qnode(dev)
        def circuit(x1, x2):
            for seq in x1:
                for gate in seq:
                    qml.apply(gate)
            for seq in x2[::-1]:
                for gate in seq[::-1]:
                    qml.adjoint(gate)

            return qml.probs(wires=range(len(self.qubits)))

        measure_probs = []
        measure_0s = []

        for batch_x1, batch_x2 in zip(gates_x1, gates_x2):
            measure_prob = circuit(batch_x1, batch_x2)
            measure_probs.append(measure_prob)
            measure_0s.append(measure_prob[0])

        return self.compute_state_stats(measure_probs), measure_0s

    def get_obs_eval(self):
        dev = qml.device("default.qubit", wires=self.qubits)

        gates_x = [list(row) for row in zip(*self.circuit_gates_x)]

        @qml.qnode(dev)
        def circuit(x):
            for seq in x:
                for gate in seq:
                    qml.apply(gate)

            return qml.probs(wires=range(len(self.qubits)))

        measure_probs = []
        for batch_x1 in gates_x:
            measure_prob = circuit(batch_x1)
            measure_probs.append(measure_prob)

        return self.compute_state_stats(measure_probs)

    def step_eval(self, action, x):
        action_gate_x = self.select_action(action, x)
        self.circuit_gates_x.append(action_gate_x)

        return self.get_obs_eval()

    def step(self, action, X1, X2, Y_batch):
        action_gate_x1 = self.select_action(action, X1)
        action_gate_x2 = self.select_action(action, X2)

        self.circuit_gates_x1.append(action_gate_x1)
        self.circuit_gates_x2.append(action_gate_x2)

        state_stats, measure_0s = self.get_obs()

        loss_fn = torch.nn.MSELoss(reduction='none')
        measure_0s = torch.stack([torch.tensor(i) for i in measure_0s])
        measure_loss = loss_fn(measure_0s, Y_batch)

        reward = 1 - measure_loss.mean()
        terminal = len(self.circuit_gates_x1) >= self.max_timesteps

        return state_stats, reward, terminal

# Function to train RL_legacy policy
def train_dqn(X_train_transformed, Y_train, policy_net, target_net, optimizer, env, num_episodes, gamma, replay_buffer, batch_size, target_update):
    policy_losses = []  # 손실 값을 저장할 리스트
    for episode in range(num_episodes):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train_transformed, Y_train)
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)  # [batch_size, state_size * 4]
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # Epsilon-greedy 정책
            epsilon = 0.01  # 에피소드에 따라 감소시키는 방법을 사용할 수 있습니다.
            if random.random() < epsilon:
                # 랜덤 행동 선택
                action = torch.randint(0, action_size, (data_size,), dtype=torch.long)  # [batch_size, num_of_qubits]
            else:
                # Q-값에 따라 행동 선택
                with torch.no_grad():
                    q_values = policy_net(state)  # [batch_size, num_of_qubits, action_size]
                    action = torch.max(q_values, dim=1)[1]  # [batch_size, num_of_qubits]


            # 환경에서 다음 상태, 보상 등 얻기
            next_state, reward, done = env.step(action.numpy(), X1_batch.numpy(), X2_batch.numpy(), Y_batch)
            # next_state = torch.tensor(next_state, dtype=torch.float32)  # [batch_size, state_size * 4]
            total_reward += reward

            # 리플레이 버퍼에 저장
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            step_count += 1

            # 일정 시간마다 학습
            if len(replay_buffer) >= batch_size:
                # 미니배치 샘플링
                transitions = replay_buffer.sample(batch_size)
                batch = Transition(*transitions)

                # 배치 데이터 처리
                state_batch = torch.stack(batch.state)  # [batch_size, state_size * 4]
                action_batch = torch.stack(batch.action)  # [batch_size, num_of_qubits]
                reward_batch = torch.tensor(batch.reward, dtype=torch.float32)  # [batch_size]
                next_state_batch = torch.stack(batch.next_state)  # [batch_size, state_size * 4]
                done_batch = torch.tensor(batch.done, dtype=torch.float32)  # [batch_size]

                # 현재 Q-값 계산
                q_values = policy_net(state_batch).permute(1,0,2)  # [batch_size, num_of_qubits, action_size]
                action_batch_expanded = action_batch.unsqueeze(-1)  # [batch_size, num_of_qubits, 1]
                state_action_values = q_values.gather(2, action_batch_expanded).squeeze(-1)  # [batch_size, num_of_qubits]

                # 타깃 Q-값 계산
                with torch.no_grad():
                    next_q_values = target_net(next_state_batch).permute(1,0,2)   # [batch_size, num_of_qubits, action_size]
                    max_next_q_values = next_q_values.max(dim=2)[0]  # [batch_size, num_of_qubits]
                    target_values = reward_batch.unsqueeze(1) + gamma * max_next_q_values * (1 - done_batch.unsqueeze(1))

                # 손실 계산
                loss_fn = nn.MSELoss()
                loss = loss_fn(state_action_values, target_values)

                # 모델 최적화
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

            # policy_losses.append(loss.item())

        # 타깃 네트워크 업데이트
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        policy_losses.append(loss.item())

        print(f'Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}')

    return policy_net, policy_losses

# Function to generate action sequence
def generate_action_sequence(policy_net, X_train_transformed, max_steps):
    env_eval = QASEnv(num_of_qubit=data_size, max_timesteps=max_steps, batch_size=batch_size)
    state, _ = env_eval.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # [1, state_size * 4]
    action_sequence = []

    for step in range(max_steps):
        with torch.no_grad():
            q_values = policy_net(state)  # [1, num_of_qubits, action_size]
            action = q_values.max(dim=2)[1]  # [1, num_of_qubits]

        action_sequence.append(action.squeeze(0).numpy())  # [num_of_qubits]

        # 다음 상태 얻기
        next_state = env_eval.step_eval(action.squeeze(0).numpy(), X_train_transformed)
        state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)  # [1, state_size * 4]

        if step % 3 == 0:
            print(f'{step + 1}/{max_steps} actions generated')

    return action_sequence



def circuit_training(X_train, Y_train, scheme, NQE_model = None, action_sequence = None):
    weights = np.random.random(30, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=QCNN_learning_rate)
    loss_history = []
    for it in range(QCNN_steps):
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]
        weights, cost_new = opt.step_and_cost(
            lambda v: cost(v, X_batch, Y_batch, scheme, NQE_model, action_sequence),
            weights)
        loss_history.append(cost_new)
        if it % 3 == 0:
            print("iteration: ", it, " cost: ", cost_new)
    return loss_history, weights


def Linear_Loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss += 0.5 * (1 - l * p)
    return loss / len(labels)

def cost(weights, X_batch, Y_batch, scheme, NQE_model = None, action_sequence = None):
    preds = []
    for x in X_batch:
        if scheme == 'NQE_RL':
            pred = QCNN_classifier(weights, x, scheme, NQE_model, action_sequence)
        elif scheme == 'NQE':
            pred = QCNN_classifier(weights, x, scheme, NQE_model)
        else:
            pred = QCNN_classifier(weights, x, scheme)
        preds.append(pred)
    return Linear_Loss(Y_batch, preds)


@qml.qnode(dev)
def QCNN_classifier(params, x, scheme, NQE_model = None, action_sequence=None):
    if scheme == 'NQE_RL':
        statepreparation(x, scheme, NQE_model, action_sequence)
    elif scheme == 'NQE':
        statepreparation(x, scheme, NQE_model)
    else:
        statepreparation(x, scheme)
    QCNN(params)
    return qml.expval(qml.PauliZ(2))


def statepreparation(x, scheme, NQE_model = None, action_sequence=None):
    if scheme is None:
        quantum_embedding_zz(x)
    elif scheme == 'NQE':
        x = NQE_model.linear_relu_stack1(torch.tensor(x, dtype=torch.float32))
        x = x.detach().numpy()
        quantum_embedding_zz(x)
    elif scheme == 'NQE_RL':
        x = NQE_model.linear_relu_stack1(torch.tensor(x, dtype=torch.float32))
        x = x.detach().numpy()
        quantum_embedding_rl(x, action_sequence)


def QCNN(params):
    param1 = params[0:15]
    param2 = params[15:30]

    U_SU4(param1, wires=[0, 1])
    U_SU4(param1, wires=[2, 3])
    U_SU4(param1, wires=[1, 2])
    U_SU4(param1, wires=[3, 0])
    U_SU4(param2, wires=[0, 2])


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


def accuracy_test(predictions, labels):
    acc = 0
    for l, p in zip(labels, predictions):
        if np.abs(l - p) < 1:
            acc = acc + 1
    return acc / len(labels)


def plot_nqe_loss(NQE_losses, iter):
    plt.figure()
    plt.plot(NQE_losses, label='NQE Loss')
    step = max(1, len(NQE_losses) // 10)
    plt.xticks(range(0, len(NQE_losses), step))
    plt.title(f'NQE Loop {iter}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/RL_legacy/result_plot/NQE_{iter}th.png')

def plot_policy_loss(policy_losses, iter):
    policy_losses_values = policy_losses
    plt.figure()
    plt.plot(policy_losses_values, color='orange',
             label='Policy Loss')
    step = max(1, len(policy_losses_values) // 10)
    plt.xticks(range(0, len(policy_losses_values), step))
    plt.title(f'Policy Loop {iter}')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/RL_legacy/result_plot/Policy_{iter}th.png')

def draw_circuit(action_sequence, iter):
    @qml.qnode(dev)
    def fig_circ(action_sequence):
        quantum_embedding_rl(np.array([1, 1, 1, 1]), action_sequence)
        return qml.probs(wires=range(4))

    if action_sequence is not None:
        fig, ax = qml.draw_mpl(fig_circ)(action_sequence)

        action_text = "\n".join(
            [str(action_sequence[i:i + 5]) for i in
             range(0, len(action_sequence), 5)]
        )
        fig.text(0.1, -0.1, f'Action Sequence: {action_text}', fontsize=8,
                 wrap=True)

        fig.savefig(f'/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/RL_legacy/result_plot/RL_circuit_{iter}th.png', bbox_inches='tight')

def plot_comparison(loss_none, loss_NQE, loss_NQE_RL,
                    accuracy_none, accuracy_NQE, accuracy_NQE_RL):
    plt.figure()
    plt.plot(loss_none, label=f'None {accuracy_none:.3f}', color='blue')
    plt.plot(loss_NQE, label=f'NQE {accuracy_NQE:.3f}', color='green')
    plt.plot(loss_NQE_RL, label=f'NQE & RL_legacy {accuracy_NQE_RL:.3f}', color='red')
    step = max(1, len(loss_none) // 10)
    plt.xticks(range(0, len(loss_none), step))
    plt.title('QCNN')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/RL_legacy/result_plot/QCNN.png')

# Main iterative process
if __name__ == "__main__":
    # Number of total iterations
    total_iterations = 2

    # Parameter settings
    data_size = 4  # Data reduction size from 256->, determine # of qubit
    batch_size = 3

    # Parameter for NQE
    N_layers = 1
    NQE_iterations = 2

    # Parameter for RL_legacy
    gamma = 0.98
    RL_learning_rate = 0.01
    state_size = data_size ** 2
    action_size = 5  # Number of possible actions, RX, RY, RZ, H, CX
    episodes = 3
    max_steps = 8

    # Parameters for QCNN
    QCNN_steps = 2
    QCNN_learning_rate = 0.01
    QCNN_batch_size = 25

    # Load data
    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='kmnist', reduction_size=data_size)

    Transition = namedtuple('Transition',
                            ('state', 'action', 'reward', 'next_state', 'done'))

    NQE_models = []
    Policy_models = []
    action_sequences = []

    for iter in range(total_iterations):
        print(f"Starting iteration {iter + 1}/{total_iterations}")
        # Step 1: Train NQE
        if iter == 0:
            action_sequence = None
        NQE_model, NQE_losses = train_NQE(X_train, Y_train, NQE_iterations, batch_size, action_sequence)
        NQE_models.append(NQE_model)

        # Step 2: Transform X_train using NQE_model
        X_train_transformed = transform_data(NQE_model, X_train)

        # Step 3: Train RL_legacy policy
        policy_net = DQNNetwork(state_size=state_size, action_size=action_size,
                                num_of_qubits=data_size)
        target_net = DQNNetwork(state_size=state_size, action_size=action_size,
                                num_of_qubits=data_size)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = torch.optim.Adam(policy_net.parameters(), lr=RL_learning_rate)
        replay_buffer = ReplayBuffer(capacity=10000)

        target_update = 10

        env = QASEnv(num_of_qubit=data_size, max_timesteps=max_steps, batch_size=batch_size)
        policy_net, policy_losses = train_dqn(
            X_train_transformed=X_train_transformed,
            Y_train=Y_train,
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            env=env,
            num_episodes=episodes,
            gamma=gamma,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            target_update=target_update
        )
        Policy_models.append(policy_net)

        # Step 4: Generate action_sequence
        action_sequence = generate_action_sequence(policy_net, X_train_transformed, max_steps)
        action_sequences.append(action_sequence)

        # save loss history fig
        plot_nqe_loss(NQE_losses, iter)
        plot_policy_loss(policy_losses, iter)
        draw_circuit(action_sequence, iter)


    # After iterations, use the final NQE model and action_sequence for QCNN
    first_NQE_model = NQE_models[0]
    final_NQE_model = NQE_models[-1]
    final_action_sequence = action_sequences[-1]

    # Convert labels for QCNN
    Y_train_QCNN = [-1 if y == 0 else 1 for y in Y_train]
    Y_test_QCNN = [-1 if y == 0 else 1 for y in Y_test]

    # Train QCNN with final NQE and action_sequence
    loss_history_with_none, weight_with_none = circuit_training(
        X_train=X_train,
        Y_train=Y_train_QCNN,
        scheme=None)
    loss_history_with_NQE, weight_with_NQE = circuit_training(
        X_train=X_train,
        Y_train=Y_train_QCNN,
        scheme="NQE",
        NQE_model=first_NQE_model)
    loss_history_with_NQE_RL, weight_with_NQE_RL = circuit_training(
        X_train=X_train,
        Y_train=Y_train_QCNN,
        scheme="NQE_RL",
        NQE_model=final_NQE_model,
        action_sequence=final_action_sequence)

    # Evaluate QCNN
    prediction_with_none = [QCNN_classifier(weight_with_none, x, None) for x in X_test]
    prediction_with_NQE = [QCNN_classifier(weight_with_NQE, x, "NQE", first_NQE_model) for x in X_test]
    prediction_with_NQE_RL = [QCNN_classifier(weight_with_NQE_RL, x, "NQE_RL", final_NQE_model, final_action_sequence) for x in X_test]

    accuracy_with_none = accuracy_test(prediction_with_none, Y_test_QCNN) * 100
    accuracy_with_NQE = accuracy_test(prediction_with_NQE, Y_test_QCNN) * 100
    accuracy_with_NQE_RL = accuracy_test(prediction_with_NQE_RL, Y_test_QCNN) * 100

    plot_comparison(loss_history_with_none, loss_history_with_NQE, loss_history_with_NQE_RL,
                    accuracy_with_none, accuracy_with_NQE, accuracy_with_NQE_RL)

    print(f"Accuracy without NQE: {accuracy_with_none:.3f}")
    print(f"Accuracy with NQE: {accuracy_with_NQE:.3f}")
    print(f"Accuracy with NQE & RL_legacy: {accuracy_with_NQE_RL:.3f}")
