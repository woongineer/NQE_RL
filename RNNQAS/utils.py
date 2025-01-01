import random

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch


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


def plot_policy_loss(arch_list):
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
    plt.savefig('loss.png')
