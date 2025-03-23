import pennylane as qml
from pennylane import numpy as np
import random

def initialize_circuit(input, types_of_circuit, layer_or_depth, num_of_qubit, gate_types = None):
    if types_of_circuit == "zz":
        return _zz_embedding(input, layer_or_depth, num_of_qubit)
    elif types_of_circuit == "random":
        if gate_types is None:
            raise ValueError("For 'random' circuit, 'gate_types' must be provided.")
        return _random_embedding(input, layer_or_depth, num_of_qubit, gate_types)
    else:
        raise ValueError("Invalid types of circuit")
    

def _random_embedding(input, layer_or_depth, num_of_qubit, gate_types):
    circuit_info = []
    for i in range(layer_or_depth):
        rand_control, rand_target = random.sample(range(num_of_qubit), 2)
        rand_gate = random.choice(gate_types)
        rand_param = random.choice(input)

        if rand_gate == "RX":
            qml.RX(rand_param, wires=rand_control)
        elif rand_gate == "RY":
            qml.RY(rand_param, wires=rand_control)
        elif rand_gate == "RZ":
            qml.RZ(rand_param, wires=rand_control)
        elif rand_gate == "CNOT":
            qml.CNOT(wires=[rand_control, rand_target])
        elif rand_gate == "H":
            qml.Hadamard(wires=rand_control)
        elif rand_gate == "I":
            qml.Identity(wires=rand_control)
        elif rand_gate == "RX_arctan":
            qml.RX(np.arctan(rand_param), wires=rand_control)
        elif rand_gate == "RY_arctan":
            qml.RY(np.arctan(rand_param), wires=rand_control)
        elif rand_gate == "RZ_arctan":
            qml.RZ(np.arctan(rand_param), wires=rand_control)
        else:
            raise ValueError("Invalid gate type")

        gate_info = {
            "gate_type": rand_gate,
            "depth": i,
            "qubits": (rand_control, rand_target),
            "param": rand_param
        }
        circuit_info.append(gate_info)

    return circuit_info

    

def _zz_embedding(input, layer_or_depth, num_of_qubit):  ##TODO 나중에 하기
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