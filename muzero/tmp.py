import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import math
import copy
import random
from collections import deque

# 액션 매핑 정의 (논문의 Table 1에 해당)
# 액션 매핑 정의 (각 n 값에 대한 변형을 개별 액션으로 추가)
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


class QuantumCircuitEnv(gym.Env):
    def __init__(self, n_qubits, n_max_depth, action_mapping, data, labels):
        super(QuantumCircuitEnv, self).__init__()
        self.n_qubits = n_qubits
        self.n_max_depth = n_max_depth
        self.action_mapping = action_mapping
        self.data = data  # 입력 데이터 X
        self.labels = labels  # 레이블 y
        self.num_actions = len(action_mapping)
        # 관찰 공간과 행동 공간 정의
        self.observation_space = spaces.Box(low=-1, high=self.num_actions - 1,
                                            shape=(self.n_max_depth,),
                                            dtype=np.int32)
        self.action_space = spaces.Discrete(self.num_actions)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.actions_taken = [-1] * self.n_max_depth
        self.done = False
        return np.array(self.actions_taken, dtype=np.int32)

    def step(self, action):
        if self.done:
            raise Exception("Episode is done")
        # 선택한 행동을 추가
        self.actions_taken[self.current_step] = action
        self.current_step += 1
        # 회로 구성
        circuit_template = self.build_circuit(self.actions_taken[:self.current_step])
        # 보상 계산
        reward = self.compute_reward(circuit_template)
        # 에피소드 완료 여부 확인
        if self.current_step >= self.n_max_depth:
            self.done = True
        else:
            self.done = False
        # 다음 관찰 반환
        obs = np.array(self.actions_taken, dtype=np.int32)
        return obs, reward, self.done, {}

    def build_circuit(self, actions):
        # PennyLane을 사용하여 회로 템플릿 생성
        def circuit(inputs):
            qml.BasisState(np.zeros(self.n_qubits), wires=range(self.n_qubits))
            for action in actions:
                if action == -1:
                    continue
                gate = self.action_mapping[action]
                self.apply_gate(gate, inputs)
        return circuit

    def apply_gate(self, gate_info, inputs):
        wires = range(self.n_qubits)
        if isinstance(gate_info, str):
            gate = gate_info
            n = None
        else:
            gate, n = gate_info  # gate는 문자열, n은 정수
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
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
        elif gate == 'CY':
            for qubit in range(self.n_qubits - 1):
                qml.CY(wires=[qubit, qubit + 1])
        elif gate == 'CZ':
            for qubit in range(self.n_qubits - 1):
                qml.CZ(wires=[qubit, qubit + 1])
        elif gate == 'CRx_pi_over_n':
            angle = np.pi / n
            for qubit in range(self.n_qubits - 1):
                qml.CRX(angle, wires=[qubit, qubit + 1])
        elif gate == 'CRy_pi_over_n':
            angle = np.pi / n
            for qubit in range(self.n_qubits - 1):
                qml.CRY(angle, wires=[qubit, qubit + 1])
        elif gate == 'CRz_pi_over_n':
            angle = np.pi / n
            for qubit in range(self.n_qubits - 1):
                qml.CRZ(angle, wires=[qubit, qubit + 1])
        elif gate == 'Rx_pi_x':
            for qubit in wires:
                qml.RX(np.pi * inputs[qubit], wires=qubit)
        elif gate == 'Ry_pi_x':
            for qubit in wires:
                qml.RY(np.pi * inputs[qubit], wires=qubit)
        elif gate == 'Rz_pi_x':
            for qubit in wires:
                qml.RZ(np.pi * inputs[qubit], wires=qubit)
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
                qml.RX(np.arctan(inputs[qubit]), wires=qubit)
        elif gate == 'Ry_arctan_x':
            for qubit in wires:
                qml.RY(np.arctan(inputs[qubit]), wires=qubit)
        elif gate == 'Rz_arctan_x':
            for qubit in wires:
                qml.RZ(np.arctan(inputs[qubit]), wires=qubit)
        elif gate == 'H':
            for qubit in wires:
                qml.Hadamard(wires=qubit)
        else:
            pass  # 필요에 따라 추가 게이트 처리

    def compute_reward(self, circuit_template):
        # QML 모델 평가하여 보상 계산
        score = self.evaluate_qml_model(circuit_template)
        reward = score
        return reward

    def evaluate_qml_model(self, circuit_template):
        # PQK를 사용하여 모델 평가
        num_samples = len(self.data)
        features = np.zeros((num_samples, self.n_qubits * 3))  # 3 Pauli 연산자 * 큐빗 수
        dev = qml.device('default.qubit', wires=self.n_qubits)

        @qml.qnode(dev)
        def circuit(inputs):
            circuit_template(inputs)
            return [qml.expval(qml.PauliX(wires=i)) for i in range(self.n_qubits)] + \
                   [qml.expval(qml.PauliY(wires=i)) for i in range(self.n_qubits)] + \
                   [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        for idx, x_i in enumerate(self.data):
            features[idx] = circuit(x_i)

        # 커널 행렬 계산
        gamma = 1.0  # 하이퍼파라미터
        kernel_matrix = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                diff = features[i] - features[j]
                sq_norm = np.sum(diff ** 2)
                kernel_matrix[i, j] = np.exp(-gamma * sq_norm)
        # SVM 학습 및 교차 검증
        clf = SVC(kernel='precomputed')
        scores = cross_val_score(clf, kernel_matrix, self.labels, cv=3)
        mean_score = np.mean(scores)
        return mean_score

# MCTS에서 사용할 노드 클래스 정의
class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = 0  # 플레이어 번호, 여기서는 단일 에이전트이므로 사용하지 않음
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

# MuZero 에이전트 클래스 정의
class MuZeroAgent(nn.Module):
    def __init__(self, observation_space, action_space, config):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        # 신경망 구성
        self.representation_network = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.dynamics_network = nn.Sequential(
            nn.Linear(128 + action_space.n, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.policy_network = nn.Linear(128, action_space.n)
        self.value_network = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])
        # 경험 저장을 위한 버퍼
        self.buffer = deque(maxlen=10000)

    def initial_inference(self, observation):
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
        hidden_state = self.representation_network(obs_tensor)
        policy_logits = self.policy_network(hidden_state)
        value = self.value_network(hidden_state)
        return hidden_state.squeeze(0), policy_logits.squeeze(0), value.squeeze(0)

    def recurrent_inference(self, hidden_state, action):
        action_one_hot = torch.zeros(self.action_space.n)
        action_one_hot[action] = 1
        x = torch.cat([hidden_state, action_one_hot.unsqueeze(0)], dim=1)
        next_hidden_state = self.dynamics_network(x)
        policy_logits = self.policy_network(next_hidden_state)
        value = self.value_network(next_hidden_state)
        return next_hidden_state.squeeze(0), policy_logits.squeeze(0), value.squeeze(0)

    def select_action(self, root):
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        actions = list(root.children.keys())
        # 방문 횟수에 비례하여 확률적으로 행동 선택
        action = np.random.choice(actions, p=visit_counts / visit_counts.sum())
        return action

    def train(self, env):
        num_episodes = self.config['num_episodes']
        for episode in range(num_episodes):
            observation = env.reset
            done = False
            total_reward = 0
            while not done:
                # MCTS 실행하여 루트 노드 생성
                root = self.run_mcts(observation)
                # 행동 선택
                action = self.select_action(root)
                # 환경에서 한 단계 진행
                next_observation, reward, done, _ = env.step(action)
                total_reward += reward
                # 경험 버퍼에 저장
                self.buffer.append((observation, action, reward))
                # 네트워크 업데이트
                if len(self.buffer) >= self.config['batch_size']:
                    self.update_network()
                # 다음 상태로 이동
                observation = next_observation
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    def run_mcts(self, observation):
        root = Node(0)
        hidden_state, policy_logits, value = self.initial_inference(observation)
        root.hidden_state = hidden_state
        # 초기 정책 확률 계산
        policy = torch.softmax(policy_logits, dim=0).detach().numpy()
        for action in range(self.action_space.n):
            root.children[action] = Node(policy[action])
        # 시뮬레이션 실행
        for _ in range(self.config['num_simulations']):
            node = root
            search_path = [node]
            # 트리 탐색
            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)
            # 확장 및 평가
            value = self.evaluate(node, action)
            # 백업
            self.backpropagate(search_path, value)
        return root

    def select_child(self, node):
        # UCB 점수 계산하여 자식 노드 선택
        max_ucb = -float('inf')
        best_action = None
        best_child = None
        for action, child in node.children.items():
            ucb = self.ucb_score(node, child)
            if ucb > max_ucb:
                max_ucb = ucb
                best_action = action
                best_child = child
        return best_action, best_child

    def ucb_score(self, parent, child):
        c_puct = self.config['c_puct']
        prior_score = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        value_score = child.value()
        return value_score + prior_score

    def evaluate(self, node, action):
        # 동적 및 예측 함수 호출
        next_hidden_state, policy_logits, value = self.recurrent_inference(node.hidden_state, action)
        node.hidden_state = next_hidden_state
        # 자식 노드 확장
        policy = torch.softmax(policy_logits, dim=0).detach().numpy()
        for action in range(self.action_space.n):
            node.children[action] = Node(policy[action])
        return value.item()

    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            # 값 업데이트
            value = node.reward + self.config['discount'] * value

    def update_network(self):
        batch = random.sample(self.buffer, self.config['batch_size'])
        observations, actions, rewards = zip(*batch)
        observations = torch.tensor(observations, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        # 신경망 예측
        hidden_states = self.representation_network(observations)
        policy_logits = self.policy_network(hidden_states)
        values = self.value_network(hidden_states).squeeze()
        # 손실 계산
        policy_loss = nn.CrossEntropyLoss()(policy_logits, actions)
        value_loss = nn.MSELoss()(values, rewards)
        loss = policy_loss + value_loss
        # 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 구성 파라미터 설정
config = {
    'lr': 0.001,
    'num_simulations': 20,
    'num_episodes': 10,
    'batch_size': 32,
    'c_puct': 1.0,
    'discount': 0.99,
}

# 메인 실행부
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 샘플 데이터 설정
    n_qubits = 2
    n_max_depth = 5
    X = np.array([[0.1, 0.2],
                  [0.3, 0.4],
                  [0.5, 0.6],
                  [0.7, 0.8]])
    y = np.array([0, 1, 0, 1])

    env = QuantumCircuitEnv(n_qubits, n_max_depth, action_mapping, X, y)
    agent = MuZeroAgent(env.observation_space, env.action_space, config)
    agent.train(env)
