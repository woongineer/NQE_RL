import gym
import pennylane as qml
import tensorflow as tf
import torch
import torch.optim as optim
from pennylane import numpy as np
from sklearn.decomposition import PCA
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드 및 전처리 함수
def data_load_and_process(dataset='mnist', reduction_size: int = 4):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
    train_filter_tf = np.where((y_train == 0) | (y_train == 1))
    test_filter_tf = np.where((y_test == 0) | (y_test == 1))

    x_train, y_train = x_train[train_filter_tf], y_train[train_filter_tf]
    x_test, y_test = x_test[test_filter_tf], y_test[test_filter_tf]

    x_train = tf.image.resize(x_train[:], (256, 1)).numpy()
    x_test = tf.image.resize(x_test[:], (256, 1)).numpy()
    x_train, x_test = np.squeeze(x_train), np.squeeze(x_test)

    X_train = PCA(reduction_size).fit_transform(x_train)
    X_test = PCA(reduction_size).fit_transform(x_test)
    x_train, x_test = [], []
    for x in X_train:
        x = (x - x.min()) * (np.pi / (x.max() - x.min()))
        x_train.append(x)
    for x in X_test:
        x = (x - x.min()) * (np.pi / (x.max() - x.min()))
        x_test.append(x)
    return x_train, x_test, y_train, y_test

# 하이브리드 모델을 위한 새로운 데이터 생성 함수
def new_data(batch_size, X, Y):
    X1_new, X2_new, Y_new = [], [], []
    for i in range(batch_size):
        n, m = np.random.randint(len(X)), np.random.randint(len(X))
        X1_new.append(X[n])
        X2_new.append(X[m])
        Y_new.append(1 if Y[n] == Y[m] else 0)

    X1_new_array = np.array(X1_new)
    X1_new_tensor = torch.from_numpy(X1_new_array).float()
    X2_new_array = np.array(X2_new)
    X2_new_tensor = torch.from_numpy(X2_new_array).float()
    Y_new_array = np.array(Y_new)
    Y_new_tensor = torch.from_numpy(Y_new_array).float()
    return X1_new_tensor, X2_new_tensor, Y_new_tensor

def state_to_tensor(state):
    pennylane_to_torch = [torch.tensor(i) for i in state]
    stacked = torch.stack(pennylane_to_torch)
    return stacked.float()

# 정책 신경망 정의
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
        adjust_action_probs = (action_probs + epsilon) / (1 + epsilon * action_probs.size(-1))

        return adjust_action_probs

# 강화학습 환경 정의
class QASEnv(gym.Env):
    def __init__(self, num_of_qubit: int = 4, max_timesteps: int = 14 * 3, batch_size: int = 1):
        super().__init__()
        self.simulator = qml.device('default.qubit', wires=num_of_qubit)
        self.qubits = self.simulator.wires.tolist()
        self.max_timesteps = max_timesteps
        self.batch_size = batch_size

    def reset(self):
        self.circuit_gates_x1 = [[] for _ in range(self.batch_size)]
        self.circuit_gates_x2 = [[] for _ in range(self.batch_size)]
        return self.get_obs(), None

    def select_action(self, action, input):
        action_set = []
        for i in range(self.batch_size):
            action_per_batch = []
            for idx, qubit in enumerate(self.qubits):
                next_qubit = self.qubits[(idx + 1) % len(self.qubits)]
                if action[i] == None:
                    action_per_batch.append(qml.Identity(wires=idx))
                elif action[i] == 0:
                    action_per_batch.append(qml.Hadamard(wires=idx))
                elif action[i] == 1:
                    action_per_batch.append(qml.RX(input[i][idx].item(), wires=idx))
                elif action[i] == 2:
                    action_per_batch.append(qml.RY(input[i][idx].item(), wires=idx))
                elif action[i] == 3:
                    action_per_batch.append(qml.RZ(input[i][idx].item(), wires=idx))
                elif action[i] == 4:
                    action_per_batch.append(qml.CNOT(wires=[qubit, next_qubit]))
            action_set.append(action_per_batch)
        return action_set

    def get_obs(self):
        # For simplicity, return dummy observation
        return [torch.zeros(4 * 4) for _ in range(self.batch_size)]

    def step(self, action, X1, X2, Y_batch):
        # For simplicity, this function does not perform actual quantum computation
        # In a real implementation, this should return the next state, reward, and done flag
        next_state = self.get_obs()
        reward = torch.zeros(self.batch_size)
        done = False
        return next_state, reward, done

# 액션 시퀀스 생성 함수
def generate_action_sequences(policy, env, X_batch, max_timesteps):
    action_sequences = []
    for x in X_batch:
        X1_new = torch.tensor([x.numpy()], dtype=torch.float32)
        X2_new = torch.tensor([x.numpy()], dtype=torch.float32)
        state, _ = env.reset()
        action_list = []
        prev_action = torch.tensor([999 for _ in range(env.batch_size)])
        timestep = 0

        while timestep < max_timesteps:
            state_tensor = state_to_tensor(state)
            prob = policy.forward(state_tensor)
            dist = torch.distributions.Categorical(prob)

            action = dist.sample()

            # 이전 액션과 동일한 액션 방지
            mask = (action == prev_action)
            while mask.any():
                new_samples = dist.sample()
                action[mask] = new_samples[mask]
                mask = (action == prev_action)
            prev_action = action

            next_state, _, _ = env.step(action, X1_new, X2_new, None)

            action_list.append(action[0].item())
            state = next_state
            timestep += 1

        action_sequences.append(action_list)
    return action_sequences

# QuantumEmbedding_RL 함수 수정
def QuantumEmbedding_RL(action_sequence, x):
    for idx, act in enumerate(action_sequence):
        if act == None:
            qml.Identity(wires=idx)
        elif act == 0:
            qml.Hadamard(wires=idx)
        elif act == 1:
            qml.RX(x[idx], wires=idx)
        elif act == 2:
            qml.RY(x[idx], wires=idx)
        elif act == 3:
            qml.RZ(x[idx], wires=idx)
        elif act == 4:
            qml.CNOT(wires=[idx, (idx + 1) % len(x)])

# Model_Fidelity 클래스 수정
class Model_Fidelity(nn.Module):
    def __init__(self):
        super(Model_Fidelity, self).__init__()
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

    def forward(self, x1_batch, x2_batch, action_sequences_x1, action_sequences_x2):
        outputs = []
        for i in range(len(x1_batch)):
            x1 = self.linear_relu_stack1(x1_batch[i])
            x2 = self.linear_relu_stack1(x2_batch[i])
            output = self.circuitRL(x1, x2, action_sequences_x1[i], action_sequences_x2[i])
            outputs.append(output)
        outputs = torch.stack(outputs)
        return outputs

    def circuitRL(self, x1, x2, action_sequence_x1, action_sequence_x2):
        dev = qml.device('default.qubit', wires=4)

        @qml.qnode(dev, interface="torch")
        def circuit():
            QuantumEmbedding_RL(action_sequence_x1, x1)
            qml.adjoint(QuantumEmbedding_RL)(action_sequence_x2, x2)
            return qml.probs(wires=range(4))

        probs = circuit()
        return probs[0]

# x_transform 모델 정의
class x_transform(nn.Module):
    def __init__(self):
        super(x_transform, self).__init__()
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

    def forward(self, x):
        return self.linear_relu_stack1(x)

# QCNN 관련 함수들
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

def QCNN_classifier(params, x):
    dev = qml.device('default.qubit', wires=4)
    @qml.qnode(dev)
    def circuit():
        for idx in range(len(x)):
            qml.RY(x[idx], wires=idx)
        QCNN(params)
        return qml.expval(qml.PauliZ(2))
    return circuit()

def Linear_Loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss += 0.5 * (1 - l * p)
    return loss / len(labels)

def cost(weights, X_batch, Y_batch):
    preds = [QCNN_classifier(weights, x) for x in X_batch]
    return Linear_Loss(Y_batch, preds)

def circuit_training(X_train, Y_train, steps, learning_rate, batch_size):
    weights = np.random.random(30, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
    loss_history = []
    for it in range(steps):
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]
        weights, cost_new = opt.step_and_cost(
            lambda v: cost(v, X_batch, Y_batch),
            weights)
        loss_history.append(cost_new)
        if it % 1 == 0:
            print("iteration: ", it+1, " cost: ", cost_new)
    return loss_history, weights

def accuracy_test(predictions, labels):
    acc = 0
    for l, p in zip(labels, predictions):
        if np.sign(p) == l:
            acc += 1
    return acc / len(labels)

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    data_size = 4
    gamma = 0.98
    learning_rate = 0.01
    state_size = data_size ** 2
    action_size = 5
    episodes = 3
    iterations = 7
    steps = 7
    batch_size = 7
    N_layers = 3

    # 데이터 로드
    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='mnist', reduction_size=data_size)

    # 정책 네트워크 및 옵티마이저 초기화
    policy = PolicyNetwork(state_size=state_size, action_size=action_size)
    optimizer_policy = optim.Adam(policy.parameters(), lr=learning_rate)

    env = QASEnv(num_of_qubit=data_size, max_timesteps=14 * N_layers, batch_size=batch_size)

    policy_losses = []

    # 강화학습 학습 루프
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
                new_samples = dist.sample()
                action[mask] = new_samples[mask]
                mask = (action == prev_action)
            prev_action = action

            next_state, reward, done = env.step(action, X1_batch, X2_batch, Y_batch)

            log_prob = dist.log_prob(action)
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

        # 정책 손실 계산
        log_probs = torch.stack(log_probs)  # Shape: [num_steps, batch_size]
        policy_loss = -log_probs * returns
        policy_loss = policy_loss.mean()
        policy_losses.append(policy_loss)

        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        print(f'E{episode + 1}/{episodes}, loss:{policy_loss}, actions:{action_list}')

        early_stop = 7
        if len(policy_losses) >= early_stop:
            last_losses = [loss.detach().item() for loss in policy_losses[-early_stop:]]
            if len(set(last_losses)) == 1:
                print('Episode early stopped')
                break

    print('Training Complete')

    # 학습된 정책 저장 및 로드
    torch.save(policy.state_dict(), 'trained_policy.pth')
    policy = PolicyNetwork(state_size=state_size, action_size=action_size)
    policy.load_state_dict(torch.load('trained_policy.pth'))
    policy.eval()

    # Model_Fidelity 학습
    model_fidelity = Model_Fidelity()
    model_fidelity.train()

    loss_fn = torch.nn.MSELoss()
    optimizer_fidelity = torch.optim.SGD(model_fidelity.parameters(), lr=0.01)

    iterations = 7
    batch_size = 5

    env = QASEnv(num_of_qubit=data_size, max_timesteps=14 * N_layers, batch_size=1)

    for it in range(iterations):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        Y_batch_binary = torch.tensor([1 if y == 1 else 0 for y in Y_batch], dtype=torch.float32)

        # 배치 내 각 데이터에 대해 액션 시퀀스 생성
        action_sequences_x1 = generate_action_sequences(policy, env, X1_batch, env.max_timesteps)
        action_sequences_x2 = generate_action_sequences(policy, env, X2_batch, env.max_timesteps)

        pred = model_fidelity(X1_batch, X2_batch, action_sequences_x1, action_sequences_x2)
        loss = loss_fn(pred, Y_batch_binary)

        optimizer_fidelity.zero_grad()
        loss.backward()
        optimizer_fidelity.step()

        if it % 1 == 0:
            print(f"Iteration: {it+1}/{iterations}, Loss: {loss.item()}")

    # x_transform 모델 학습
    model_transform = x_transform()
    optimizer_transform = torch.optim.Adam(model_transform.parameters(), lr=0.01)

    epochs = 5
    batch_size_transform = 5

    for epoch in range(epochs):
        X1_batch, X2_batch, Y_batch = new_data(batch_size_transform, X_train, Y_train)
        Y_batch_binary = torch.tensor([1 if y == 1 else 0 for y in Y_batch], dtype=torch.float32)

        # 배치 내 각 데이터에 대해 액션 시퀀스 생성
        action_sequences_x1 = generate_action_sequences(policy, env, X1_batch, env.max_timesteps)
        action_sequences_x2 = generate_action_sequences(policy, env, X2_batch, env.max_timesteps)

        # x_transform 모델을 사용하여 데이터 변환
        X1_transformed = model_transform(X1_batch)
        X2_transformed = model_transform(X2_batch)

        # 변환된 데이터를 사용하여 Model_Fidelity를 통해 예측
        pred = model_fidelity(X1_transformed, X2_transformed, action_sequences_x1, action_sequences_x2)
        loss = loss_fn(pred, Y_batch_binary)

        optimizer_transform.zero_grad()
        loss.backward()
        optimizer_transform.step()

        if epoch % 1 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Transform Loss: {loss.item()}")

    # 변환된 데이터 생성
    X_train_transformed = [model_transform(torch.tensor(x, dtype=torch.float32)).detach().numpy() for x in X_train]
    X_test_transformed = [model_transform(torch.tensor(x, dtype=torch.float32)).detach().numpy() for x in X_test]

    # QCNN 학습 및 평가
    Y_train_labels = [-1 if y == 0 else 1 for y in Y_train]
    Y_test_labels = [-1 if y == 0 else 1 for y in Y_test]

    steps = 5
    learning_rate = 0.01
    batch_size = 25

    loss_history, weights = circuit_training(X_train_transformed, Y_train_labels, steps, learning_rate, batch_size)

    # 정확도 테스트
    predictions = [QCNN_classifier(weights, x) for x in X_test_transformed]
    accuracy = accuracy_test(predictions, Y_test_labels) * 100

    print(f"Accuracy with transformed x': {accuracy:.3f}%")

    # 손실 그래프 출력
    plt.rcParams['figure.figsize'] = [10, 5]
    fig, ax = plt.subplots()
    clrs = sns.color_palette("husl", 1)
    with sns.axes_style("darkgrid"):
        ax.plot(range(len(loss_history)), loss_history, label="With transformed x'", c=clrs[0])

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("QCNN Loss History with transformed x'")
    ax.legend()

    plt.show()
