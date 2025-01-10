import random

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from data import new_data


def generate_layers(num_qubit, num_layers):
    """
    Generate a set of random layers based on predefined rules.
    Args:
        num_qubits: Number of qubits in the circuit.
        num_layers: Number of layers to generate.
    Returns:
        layer_dict: A dictionary mapping layer index to layer composition.
    """
    if num_qubit % 2 != 0:
        raise ValueError("Number of qubits must be even for this rule.")

    single_qubit_gates = ["R_x", "R_y", "R_z"]
    layer_dict = {}
    generated_layers = set()  # To track unique layers

    for layer_idx in range(num_layers):
        while True:
            # Step 1: Randomly assign single-qubit gates
            single_gates = []
            qubits_for_single = random.sample(range(num_qubit), num_qubit // 2)
            for qubit in qubits_for_single:
                gate = random.choice(single_qubit_gates)
                single_gates.append((gate, qubit))

            # Step 2: Randomly assign CNOT gates
            cnot_gates = []
            qubits_for_cnot = random.sample(range(num_qubit), num_qubit)
            for i in range(0, len(qubits_for_cnot), 2):  # Pair qubits for CNOT
                control, target = qubits_for_cnot[i], qubits_for_cnot[i + 1]
                cnot_gates.append(("CNOT", (control, target)))

            # Step 3: Combine single and CNOT gates
            layer = single_gates + cnot_gates

            # Step 4: Ensure layer is functionally unique
            layer_tuple = tuple(sorted(layer))  # Sort for uniqueness
            if layer_tuple not in generated_layers:
                generated_layers.add(layer_tuple)
                layer_dict[layer_idx] = layer
                break  # Exit the loop once a unique layer is found

    return layer_dict


def make_arch(layer_list_flat, num_qubit):
    arch = np.zeros((1, len(layer_list_flat), num_qubit, 5))
    for time, (gate, qubit_idx) in enumerate(layer_list_flat):
        if gate == 'R_x':
            arch[0, time, qubit_idx, 0] = 1
        elif gate == 'R_y':
            arch[0, time, qubit_idx, 1] = 1
        elif gate == 'R_z':
            arch[0, time, qubit_idx, 2] = 1
        elif gate == 'CNOT':
            arch[0, time, qubit_idx[0], 3] = 1
            arch[0, time, qubit_idx[1], 4] = 1

    return torch.from_numpy(arch).float()


def make_arch_sb3(layer_list_flat, num_qubit, max_layer_step, num_gate_class):
    arch = torch.zeros((len(layer_list_flat), num_qubit, num_gate_class))
    for time, (gate, qubit_idx) in enumerate(layer_list_flat):
        if gate == 'R_x':
            arch[time, qubit_idx, 0] = 1
        elif gate == 'R_y':
            arch[time, qubit_idx, 1] = 1
        elif gate == 'R_z':
            arch[time, qubit_idx, 2] = 1
        elif gate == 'CNOT':
            arch[time, qubit_idx[0], 3] = 1
            arch[time, qubit_idx[1], 4] = 1

    padded_arch = torch.zeros(max_layer_step * 4, num_qubit, num_gate_class)
    padded_arch[:arch.shape[0], :arch.shape[1], :arch.shape[2]] = arch

    return padded_arch


def quantum_embedding(x, gate_list):
    for gate, qubit_idx in gate_list:
        if gate == 'R_x':
            qml.RX(x[qubit_idx], wires=qubit_idx)
        elif gate == 'R_y':
            qml.RY(x[qubit_idx], wires=qubit_idx)
        elif gate == 'R_z':
            qml.RZ(x[qubit_idx], wires=qubit_idx)
        elif gate == 'CNOT':
            qml.CNOT(wires=[qubit_idx[0], qubit_idx[1]])


def plot_policy_loss(arch_list, filename):
    x = list(arch_list.keys())
    policy_losses = [arch_list[i]['policy_loss'] for i in x]
    NQE_losses = [arch_list[i]['NQE_loss'] for i in x]

    plt.figure(figsize=(10, 6))

    plt.plot(x, policy_losses, marker='o', linestyle='-', label='Policy Loss')
    plt.plot(x, NQE_losses, marker='s', linestyle='--', label='NQE Loss')

    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('Policy & NQE Loss')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # 기준선
    plt.legend()
    plt.grid()
    plt.savefig(filename)


def set_done_loss(max_layer_step, num_qubit, max_epoch_NQE, batch_size, X_train, Y_train, X_test, Y_test):
    dev = qml.device('default.qubit', wires=num_qubit)
    def exp_Z(x, wires):
        qml.RZ(-2 * x, wires=wires)

    # exp(i(pi - x1)(pi - x2)ZZ) gate
    def exp_ZZ2(x1, x2, wires):
        qml.CNOT(wires=wires)
        qml.RZ(-2 * (np.pi - x1) * (np.pi - x2), wires=wires[1])
        qml.CNOT(wires=wires)

    # Quantum Embedding 1 for model 1 (Conventional ZZ feature embedding)
    def QuantumEmbedding(input):
        for i in range(int(max_layer_step/5)):
            for j in range(4):
                qml.Hadamard(wires=j)
                exp_Z(input[j], wires=j)
            for k in range(3):
                exp_ZZ2(input[k], input[k + 1], wires=[k, k + 1])
            exp_ZZ2(input[3], input[0], wires=[3, 0])

    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        QuantumEmbedding(inputs[0:4])
        qml.adjoint(QuantumEmbedding)(inputs[4:8])
        return qml.probs(wires=range(4))

    class Model_Fidelity(torch.nn.Module):
        def __init__(self):
            super().__init__()
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

    model = Model_Fidelity()
    model.train()

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    for it in range(max_epoch_NQE):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        pred = model(X1_batch, X2_batch)
        loss = loss_fn(pred, Y_batch)

        opt.zero_grad()
        loss.backward()
        opt.step()

    valid_loss_list = []
    model.eval()
    for _ in range(batch_size):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_test, Y_test)
        with torch.no_grad():
            pred = model(X1_batch, X2_batch)
        valid_loss_list.append(loss_fn(pred, Y_batch))

    soft_condition = (sum(valid_loss_list) / batch_size).detach().item()
    hard_condition = min(valid_loss_list).detach().item()
    print(f'Set done standard with zz feature map setting, mean_loss:{soft_condition}, min_loss:{hard_condition}')
    return soft_condition, hard_condition