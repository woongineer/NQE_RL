import numpy as np
import cirq
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from gym.utils import seeding
from typing import List



def get_bell_state() -> np.ndarray:
    target = np.zeros(2 ** 2, dtype=complex)
    target[0] = 1. / np.sqrt(2) + 0.j
    target[-1] = 1. / np.sqrt(2) + 0.j
    return target


# Quantum Architecture Search Environment
class QuantumArchSearchEnv(gym.Env):
    def __init__(
            self,
            target: np.ndarray = get_bell_state(),
            qubits: List[cirq.LineQubit] = None,
            state_observables: List[cirq.GateOperation] = None,
            action_gates: List[cirq.GateOperation] = None,
            fidelity_threshold: float = 0.95,
            reward_penalty: float = 0.01,
            max_timesteps: int = 20,
    ):
        super().__init__()
        self.target = target
        if qubits is None:
            qubits = cirq.LineQubit.range(2)
        self.qubits = qubits
        # 뭐로 Measure 할지 XYZ
        if state_observables is None:
            state_observables = []
            for qubit in qubits:
                state_observables += [
                    cirq.X(qubit),
                    cirq.Y(qubit),
                    cirq.Z(qubit),
                ]
        self.state_observables = state_observables
        if action_gates is None:
            action_gates = []
            for idx, qubit in enumerate(qubits):
                next_qubit = qubits[(idx + 1) % len(qubits)]
                action_gates += [
                    cirq.rz(np.pi / 4.)(qubit),
                    cirq.X(qubit),
                    cirq.Y(qubit),
                    cirq.Z(qubit),
                    cirq.H(qubit),
                    cirq.CNOT(qubit, next_qubit)
                ]
        self.action_gates = action_gates
        self.fidelity_threshold = fidelity_threshold
        self.reward_penalty = reward_penalty
        self.max_timesteps = max_timesteps
        self.simulator = cirq.Simulator()

    def reset(self):
        self.circuit_gates = []
        return self._get_obs()

    def _get_cirq(self):
        circuit = cirq.Circuit(cirq.I(qubit) for qubit in self.qubits)
        for gate in self.circuit_gates:
            circuit.append(gate)
        return circuit

    def _get_obs(self):
        circuit = self._get_cirq()
        obs = self.simulator.simulate_expectation_values(circuit,
                                                         observables=self.state_observables)
        return np.array(obs).real

    def _get_fidelity(self):
        circuit = self._get_cirq()
        pred = self.simulator.simulate(circuit).final_state_vector
        inner = np.inner(np.conj(pred), self.target)
        fidelity = np.conj(inner) * inner
        return fidelity.real

    def step(self, action):
        action_gate = self.action_gates[action]
        self.circuit_gates.append(action_gate)
        observation = self._get_obs()
        fidelity = self._get_fidelity()
        reward = fidelity - self.reward_penalty if fidelity > self.fidelity_threshold else -self.reward_penalty
        terminal = (reward > 0.) or (
                len(self.circuit_gates) >= self.max_timesteps)
        info = {'fidelity': fidelity, 'circuit': self._get_cirq()}
        return observation, reward, terminal, info


# Neural Network Definitions
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 7)
        self.fc2 = nn.Linear(7, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


# Main script logic
if __name__ == "__main__":
    # Parameters
    fidelity_threshold = 0.95
    reward_penalty = 0.01
    max_timesteps = 20
    num_episodes = 3
    gamma = 0.99

    # Environment
    env = QuantumArchSearchEnv(
        target=get_bell_state(),
        fidelity_threshold=fidelity_threshold,
        reward_penalty=reward_penalty,
        max_timesteps=max_timesteps
    )

    # Neural Networks
    input_dim = len(env.state_observables)
    output_dim = env.action_space.n

    policy_net = PolicyNetwork(input_dim, output_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

    # Train
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        log_probs = []
        rewards = []

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = policy_net(state_tensor)
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

        print(f'Episode {episode + 1}/{num_episodes} complete.')

    print('Training Complete')
