import gym
import pennylane as qml
import torch
from pennylane import numpy as np
from torch import nn
import random
import matplotlib.pyplot as plt

from data import new_data, data_load_and_process

# Set your device
dev = qml.device('default.qubit', wires=4)


def exp_Z(x, wires):
    qml.RZ(-2 * x, wires=wires)  ##TODO 왜 이걸 없애니까...???????


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


def quantum_embedding_zz_new(input):
    for i in range(N_layers):
        for j in range(4):
            qml.Hadamard(wires=j)
            qml.RZ(-2 * input[j], wires=j)

        for k in range(3):
            qml.CNOT(wires=[k, k + 1])
            qml.RZ(-2 * (np.pi - input[k]) * (np.pi - input[k + 1]),
                   wires=k + 1)
            qml.CNOT(wires=[k, k + 1])

        qml.CNOT(wires=[3, 0])
        qml.RZ(-2 * (np.pi - input[3]) * (np.pi - input[0]), wires=0)
        qml.CNOT(wires=[3, 0])


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
#
# def quantum_embedding_rl(x, action_sequence):
#     for action in action_sequence:
#         for qubit_idx in range(data_size):
#             if action[qubit_idx] == 0:
#                 qml.Hadamard(wires=qubit_idx)
#             elif action[qubit_idx] == 1:
#                 qml.RX(-2 * x[qubit_idx], wires=qubit_idx)
#             elif action[qubit_idx] == 2:
#                 qml.RY(-2 * x[qubit_idx], wires=qubit_idx)
#             elif action[qubit_idx] == 3:
#                 qml.RZ(-2 * x[qubit_idx], wires=qubit_idx)
#             elif action[qubit_idx] == 4:
#                 qml.CNOT(wires=[qubit_idx, (qubit_idx + 1) % data_size])
#             elif action[qubit_idx] == 1:
#                 qml.RX(-2 * (np.pi - x[qubit_idx]), wires=qubit_idx)
#             elif action[qubit_idx] == 2:
#                 qml.RY(-2 * (np.pi - x[qubit_idx]), wires=qubit_idx)
#             elif action[qubit_idx] == 3:
#                 qml.RZ(-2 * (np.pi - x[qubit_idx]), wires=qubit_idx)


def quantum_embedding_rl_inverse(x, action_sequence):
    # Reverse the order of the action sequence
    for action in reversed(action_sequence):
        # Reverse the order of qubits
        for qubit_idx in reversed(range(data_size)):
            if action[qubit_idx] == 0:
                # Hadamard is its own inverse
                qml.Hadamard(wires=qubit_idx)
            elif action[qubit_idx] == 1:
                # Inverse of RX is RX with negative angle
                qml.RX(-x[qubit_idx], wires=qubit_idx)
            elif action[qubit_idx] == 2:
                qml.RY(-x[qubit_idx], wires=qubit_idx)
            elif action[qubit_idx] == 3:
                qml.RZ(-x[qubit_idx], wires=qubit_idx)
            elif action[qubit_idx] == 4:
                # CNOT is its own inverse
                qml.CNOT(wires=[qubit_idx, (qubit_idx + 1) % data_size])


def stater(strat):
    plt.clf()
    results = []
    for _ in range(30):
        NQE_model = NQEModel(tick=strat, action_sequence=action_sequence)
        NQE_model.train()
        results.append(NQE_model(X1_batch, X2_batch))
    results = torch.cat(results).flatten()
    plt.hist(results.detach().numpy(), bins=50, range=(0, 1))
    plt.savefig(f"{strat}.png")
    plt.clf()
    plt.close()



# Define the NQE Model
class NQEModel(torch.nn.Module):
    def __init__(self, tick=None, action_sequence=None,):
        super().__init__()
        self.action_sequence = action_sequence
        if tick == 'ori':
            # Use quantum_embedding_zz
            @qml.qnode(dev, interface="torch")
            def circuit(inputs):
                quantum_embedding_zz(inputs[0:4])
                qml.adjoint(quantum_embedding_zz)(inputs[4:8])
                return qml.probs(wires=range(4))
        elif tick == 'new':
            @qml.qnode(dev, interface="torch")
            def circuit(inputs):
                quantum_embedding_zz_new(inputs[0:4])
                qml.adjoint(quantum_embedding_zz_new)(inputs[4:8])
                return qml.probs(wires=range(4))

        elif tick == 'ori_r':
            @qml.qnode(dev, interface="torch")
            def circuit(inputs):
                quantum_embedding_rl(inputs[0:4], self.action_sequence)
                qml.adjoint(quantum_embedding_rl)(inputs[4:8],  ##TODO 되는거 맞나?
                                             self.action_sequence)
                return qml.probs(wires=range(4))
        elif tick == 'new_r':
            # Use quantum_embedding_rl
            @qml.qnode(dev, interface="torch")
            def circuit(inputs):
                quantum_embedding_rl(inputs[0:4], self.action_sequence)
                quantum_embedding_rl_inverse(inputs[4:8],  ##TODO 되는거 맞나?
                                                  self.action_sequence)
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

#
# class NQEModel(torch.nn.Module):
#     def __init__(self, action_sequence=None):
#         super().__init__()
#         self.action_sequence = action_sequence
#         if action_sequence is None:
#             # Use quantum_embedding_zz
#             @qml.qnode(dev, interface="torch")
#             def circuit(inputs):
#                 quantum_embedding_zz(inputs[..., 0:4])
#                 qml.adjoint(quantum_embedding_zz)(inputs[..., 4:8])
#                 return qml.probs(wires=range(4))
#         else:
#             # Use quantum_embedding_rl
#             @qml.qnode(dev, interface="torch")
#             def circuit(inputs):
#                 quantum_embedding_rl(inputs[0:4], self.action_sequence)
#                 qml.adjoint(quantum_embedding_rl)(inputs[4:8],  ##TODO 되는거 맞나?
#                                                   self.action_sequence)
#
#                 return qml.probs(wires=range(4))
#         self.circuit = circuit
#         self.linear_relu_stack1 = nn.Sequential(
#             nn.Linear(4, 8),
#             nn.ReLU(),
#             nn.Linear(8, 8),
#             nn.ReLU(),
#             nn.Linear(8, 4)
#         )
#
#     def forward(self, x1, x2):
#         x1 = self.linear_relu_stack1(x1)
#         x2 = self.linear_relu_stack1(x2)
#         x = torch.concat([x1, x2], 1)
#
#         result = [self.circuit(x_batch) for x_batch in x]
#         return torch.stack(result)[:, 0]


# Function to train NQE
def train_NQE(X_train, Y_train, NQE_iterations, batch_size,
              action_sequence=None):

    NQE_model = NQEModel(tick='ori')
    # NQE_model = NQEModel(tick='new')
    # NQE_model = NQEModel(tick='ori_r', action_sequence=action_sequence)
    # NQE_model = NQEModel(tick='new_r', action_sequence=action_sequence)

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


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, num_of_qubit):
        super(PolicyNetwork, self).__init__()
        self.state_linear_relu_stack = nn.Sequential(
            nn.Linear(state_size * 4, state_size * 8),
            nn.ReLU(),
            nn.Linear(state_size * 8, state_size * 4),
        )
        # qubit 별로 다른 model 적용하기
        self.action_select = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(state_size * 4, action_size * 2),
                nn.ReLU(),
                nn.Linear(action_size * 2, action_size),
            ) for _ in range(num_of_qubit)]
        )

    def forward(self, state):
        state_new = self.state_linear_relu_stack(state)

        action_probs = []
        epsilon = 0.03

        for qubit_action_select in self.action_select:
            action_prob = torch.softmax(qubit_action_select(state_new), dim=-1)
            adjust_action_probs = (action_prob + epsilon) / (
                    1 + epsilon * action_size)
            action_probs.append(adjust_action_probs)

        return torch.stack(action_probs, dim=0)


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

        loss_fn = torch.nn.MSELoss(reduction='none')  # TODO Need discussion
        measure_0s = torch.stack([torch.tensor(i) for i in measure_0s])
        measure_loss = loss_fn(measure_0s, Y_batch)

        reward = 1 - measure_loss.mean()  # TODO 개별 배치마다 따로 prob을 뽑은게 아니니까 reward도 통합해야 할 듯, measure_loss를 minimize하는게 목적이니 작을수록 큰 reward
        terminal = len(self.circuit_gates_x1) >= self.max_timesteps

        return state_stats, reward, terminal


# Function to train RL_legacy policy
def train_policy(X_train_transformed, Y_train, policy, optimizer, env, episodes,
                 gamma):
    policy_losses = []
    for episode in range(episodes):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train_transformed,
                                               Y_train)
        state, _ = env.reset()
        done = False
        log_probs = []
        rewards = []
        prev_action = torch.tensor([999 for _ in range(data_size)])

        while not done:
            prob = policy.forward(state)
            dist = torch.distributions.Categorical(prob)
            action = dist.sample()
            mask = (action == prev_action)
            while mask.any():
                new_samples = dist.sample()
                action[mask] = new_samples[mask]
                mask = (action == prev_action)
            prev_action = action

            next_state, reward, done = env.step(action, X1_batch.numpy(),
                                                X2_batch.numpy(), Y_batch)
            log_prob = dist.log_prob(action.clone().detach())
            log_probs.append(log_prob.sum())
            rewards.append(reward)
            state = next_state

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        log_probs = torch.stack(log_probs)
        policy_loss = -log_probs * returns
        policy_loss = policy_loss.mean()
        policy_losses.append(policy_loss)

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        print(f'Episode {episode + 1}/{episodes}, Loss: {policy_loss.item()}')
    return policy, policy_losses


# Function to generate action sequence
def generate_action_sequence(policy_model, X_train_transformed, max_steps):
    env_eval = QASEnv(num_of_qubit=data_size, max_timesteps=max_steps,
                      batch_size=batch_size)
    state, _ = env_eval.reset()
    action_sequence = []
    prev_action = torch.tensor(
        [999 for _ in range(env_eval.simulator.num_wires)])

    for i in range(max_steps):
        with torch.no_grad():
            prob = policy_model(state)
            dist = torch.distributions.Categorical(prob)
            action = dist.sample()
            mask = (action == prev_action)
            while mask.any():
                new_samples = dist.sample()
                action[mask] = new_samples[mask]
                mask = (action == prev_action)
            prev_action = action
        action_sequence.append(action.numpy())
        state = env_eval.step_eval(action, X_train_transformed)
        if i % 3 == 0:
            print(f'{i}/{max_steps} actions generated')

    return action_sequence


def circuit_training(X_train, Y_train, scheme, NQE_model=None,
                     action_sequence=None):
    weights = np.random.random(30, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=QCNN_learning_rate)
    loss_history = []
    for it in range(QCNN_steps):
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]
        weights, cost_new = opt.step_and_cost(
            lambda v: cost(v, X_batch, Y_batch, scheme, NQE_model,
                           action_sequence),
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


def cost(weights, X_batch, Y_batch, scheme, NQE_model=None,
         action_sequence=None):
    preds = []
    for x in X_batch:
        if scheme == 'NQE_RL':
            pred = QCNN_classifier(weights, x, scheme, NQE_model,
                                   action_sequence)
        elif scheme == 'NQE':
            pred = QCNN_classifier(weights, x, scheme, NQE_model)
        else:
            pred = QCNN_classifier(weights, x, scheme)
        preds.append(pred)
    return Linear_Loss(Y_batch, preds)


@qml.qnode(dev)
def QCNN_classifier(params, x, scheme, NQE_model=None, action_sequence=None):
    if scheme == 'NQE_RL':
        statepreparation(x, scheme, NQE_model, action_sequence)
    elif scheme == 'NQE':
        statepreparation(x, scheme, NQE_model)
    else:
        statepreparation(x, scheme)
    QCNN(params)
    return qml.expval(qml.PauliZ(2))


def statepreparation(x, scheme, NQE_model=None, action_sequence=None):
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
    plt.savefig(
        f'/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/RL_legacy/result_plot/NQE_{iter}th.png')


def plot_policy_loss(policy_losses, iter):
    policy_losses_values = [loss.item() for loss in policy_losses]
    plt.figure()
    plt.plot(policy_losses_values, color='orange',
             label='Policy Loss')
    step = max(1, len(policy_losses_values) // 10)
    plt.xticks(range(0, len(policy_losses_values), step))
    plt.title(f'Policy Loop {iter}')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(
        f'/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/RL_legacy/result_plot/Policy_{iter}th.png')


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

        fig.savefig(
            f'/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/RL_legacy/result_plot/RL_circuit_{iter}th.png',
            bbox_inches='tight')


def plot_comparison(loss_none, loss_NQE, loss_NQE_RL,
                    accuracy_none, accuracy_NQE, accuracy_NQE_RL):
    plt.figure()
    plt.plot(loss_none, label=f'None {accuracy_none:.3f}', color='blue')
    plt.plot(loss_NQE, label=f'NQE {accuracy_NQE:.3f}', color='green')
    plt.plot(loss_NQE_RL, label=f'NQE & RL_legacy {accuracy_NQE_RL:.3f}',
             color='red')
    step = max(1, len(loss_none) // 10)
    plt.xticks(range(0, len(loss_none), step))
    plt.title('QCNN')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(
        f'/Users/jwheo/Desktop/Y/NQE/Neural-Quantum-Embedding/RL_legacy/result_plot/QCNN.png')


# Main iterative process
if __name__ == "__main__":
    # Parameter settings
    data_size = 4  # Data reduction size from 256->, determine # of qubit
    batch_size = 25

    # Parameter for NQE
    N_layers = 1
    NQE_iterations = 100

    # Load data
    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='mnist',
                                                             reduction_sz=data_size)

    action_sequence = [[random.choice(range(5)) for _ in range(4)] for _ in range(15)]

    # Step 1: Train NQE
    NQE_model, NQE_losses = train_NQE(X_train, Y_train, NQE_iterations,
                                      batch_size, action_sequence)

    # save loss history fig
    plot_nqe_loss(NQE_losses, iter)
    draw_circuit(action_sequence, iter)
