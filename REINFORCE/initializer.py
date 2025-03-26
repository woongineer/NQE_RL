import random

import pennylane as qml
from pennylane import numpy as np


def initialize_circuit(types_of_circuit, depth, num_of_qubit, gate_types=None):
    if types_of_circuit == "zz":
        return _zz_embedding(depth, num_of_qubit)
    elif types_of_circuit == "random":
        if gate_types is None:
            raise ValueError("For 'random' circuit, 'gate_types' must be provided.")
        return _random_embedding(depth, num_of_qubit, gate_types)
    else:
        raise ValueError("Invalid types of circuit")


def _random_embedding(depth, num_of_qubit, gate_types):
    circuit_info = []
    for i in range(depth):
        rand_gate = random.choice(gate_types)
        rand_param_index = random.choice(range(num_of_qubit))
        if rand_gate == 'CNOT':
            qubits = tuple(random.sample(range(num_of_qubit), 2))
        else:
            qubits = (random.choice(range(num_of_qubit)), None)

        gate_info = {
            "gate_type": rand_gate,
            "depth": i,
            "qubits": qubits,
            "param": rand_param_index
        }
        circuit_info.append(gate_info)

    return circuit_info


def _zz_embedding(layer_or_depth, num_of_qubit):  ##TODO 나중에 하기
    circuit_info = []
    for i in range(layer_or_depth):
        for j in range(num_of_qubit):
            qml.Hadamard(wires=j)
            qml.RZ(-input[j], wires=j)
        for k in range(3):
            qml.CNOT(wires=[k, k + 1])
            qml.RZ(-1 * (np.pi - input[k]) * (np.pi - input[k + 1]), wires=k + 1)
            qml.CNOT(wires=[k, k + 1])

        qml.CNOT(wires=[3, 0])
        qml.RZ(-1 * (np.pi - input[3]) * (np.pi - input[0]), wires=0)
        qml.CNOT(wires=[3, 0])
