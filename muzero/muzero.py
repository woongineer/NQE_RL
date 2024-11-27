import random
from collections import deque

import gym
import numpy as np
import pennylane as qml
import torch
from pennylane import numpy as np
from torch import nn

from data import data_load_and_process as dataprep
from data import new_data

# Set your device
dev = qml.device('default.qubit', wires=4)

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



class QASEnv(gym.Env):
    def __init__(self, n_qubits, max_step):
        super().__init__()
        self.n_qubits = n_qubits
        self.max_step = max_step

        self.observation_space = gym.spaces.Box(low=-1,
                                                high=len(action_mapping) - 1,
                                                shape=(self.max_step,),
                                                dtype=np.int32)
        self.action_space = gym.spaces.Discrete(len(action_mapping))

    def reset(self):  ##TODO data도?
        self.current_step = 0
        self.actions_taken = [-1] * self.max_step
        self.done = False
        return np.array(self.actions_taken, dtype=np.int32)

    def build_circuit(self, actions, input):
        qml.BasisState(np.zeros(self.n_qubits), wires=range(self.n_qubits))
        for action in actions:
            if action == -1:
                continue
            gate_info = action_mapping[action]
            self.apply_gate(gate_info, input)

    def apply_gate(self, gate_info, input):
        wires = range(self.n_qubits)
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
                qml.RX(np.pi * input[qubit], wires=qubit)
        elif gate == 'Ry_pi_x':
            for qubit in wires:
                qml.RY(np.pi * input[qubit], wires=qubit)
        elif gate == 'Rz_pi_x':
            for qubit in wires:
                qml.RZ(np.pi * input[qubit], wires=qubit)
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
                qml.RX(np.arctan(input[qubit]), wires=qubit)
        elif gate == 'Ry_arctan_x':
            for qubit in wires:
                qml.RY(np.arctan(input[qubit]), wires=qubit)
        elif gate == 'Rz_arctan_x':
            for qubit in wires:
                qml.RZ(np.arctan(input[qubit]), wires=qubit)
        elif gate == 'H':
            for qubit in wires:
                qml.Hadamard(wires=qubit)

    def step(self, action, x_data, y_data):
        self.action = action
        if self.done:
            raise Exception("Episode is done")
        # 선택한 행동을 추가
        self.actions_taken[self.current_step] = action
        self.current_step += 1
        # 보상 계산
        reward = self.get_fidelity(x_data, y_data)
        # 에피소드 완료 여부 확인
        if self.current_step >= self.max_step:
            self.done = True
        else:
            self.done = False
        # 다음 관찰 반환
        obs = np.array(self.actions_taken, dtype=np.int32)
        return obs, reward, self.done

    @qml.qnode(dev)
    def circuit(self, x_data):
        self.build_circuit(x_data[0:4], self.action)
        qml.adjoint(self.build_circuit)(x_data[4:8], self.action)
        return qml.probs(wires=range(self.n_qubits))

    def get_fidelity(self, x_data, y_data):
        qlayer = qml.qnn.TorchLayer(self.circuit, weight_shapes={})
        pred = qlayer(x_data)[:, 0]
        return torch.nn.MSELoss()(pred, y_data)


class MuZeroAgent(nn.Module):
    def __int__(self, observation_space, action_space, config):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config

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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'])
        self.buffer = deque(config['buffer_size'])

    def initial_inference(self, observation):
        observation_tensor = torch.from_numpy(observation).float().unsqueeze(0)
        hidden_state = self.representation_network(observation_tensor)

        policy_logits = self.policy_network(hidden_state)
        value = self.value_network(hidden_state)

        return hidden_state.squeeze(0), policy_logits.squeeze(0), value.squeeze(0)



if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    episodes = 10
    batch_size = 25

    config = {
        'n_qubits': 4,
        'lr': 0.001,
        'num_simulations': 20,
        'num_episodes': 10,
        'batch_size': 32,
        'c_puct': 1.0,
        'discount': 0.99,
    }

    X_train, X_test, Y_train, Y_test = dataprep(dataset='kmnist',
                                                reduction_sz=config['n_qubits'])
    env = QASEnv(n_qubits=config['n_qubits'], max_step=8)
    agent = MuZeroAgent(observation_space=env.observation_space,
                        action_space=env.action_space, config=config)

    for episode in range(episodes):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)

        observation = env.reset()
        done = False
        total_reward = 0
        while not done:

