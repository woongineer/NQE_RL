import gym
import pennylane as qml
import torch


class QASEnv(gym.Env):
    def __init__(self,
                 num_of_qubit: int = 4,
                 max_timesteps: int = 14 * 3,  # N_layers == 3
                 batch_sz: int = 25
                 ):
        super().__init__()
        self.simulator = qml.device('default.qubit', wires=num_of_qubit)
        self.qubits = self.simulator.wires.tolist()
        self.max_timesteps = max_timesteps
        self.batch_sz = batch_sz

    def reset(self):
        dummy_action = [None for _ in range(len(self.qubits))]
        dummy_input = [[0 for _ in self.qubits] for _ in range(self.batch_sz)]

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
