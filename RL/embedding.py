import gym
import matplotlib.pyplot as plt
import pennylane as qml
import seaborn as sns
import tensorflow as tf
import torch
from pennylane import numpy as np
from sklearn.decomposition import PCA
from torch import nn
from torch.optim.lr_scheduler import StepLR


def exp_Z(x, wires):
    qml.RZ(-2 * x, wires=wires)


# exp(i(pi - x1)(pi - x2)ZZ) gate
def exp_ZZ2(x1, x2, wires):
    qml.CNOT(wires=wires)
    qml.RZ(-2 * (np.pi - x1) * (np.pi - x2), wires=wires[1])
    qml.CNOT(wires=wires)

def quantum_embedding_zz(input):
    N_layers = 1
    for i in range(N_layers):
        for j in range(4):
            qml.Hadamard(wires=j)
            exp_Z(input[j], wires=j)
        for k in range(3):
            exp_ZZ2(input[k], input[k + 1], wires=[k, k + 1])
        exp_ZZ2(input[3], input[0], wires=[3, 0])

def quantum_embedding_rl(x, action_sequence):
    for action in action_sequence:
        for qubit_idx in range(len(x)):  #len(x)==data_size, 어떻게 끌어올지 모르겠음
            if action[qubit_idx] == 0:
                qml.Hadamard(wires=qubit_idx)
            elif action[qubit_idx] == 1:
                qml.RX(x[qubit_idx], wires=qubit_idx)
            elif action[qubit_idx] == 2:
                qml.RY(x[qubit_idx], wires=qubit_idx)
            elif action[qubit_idx] == 3:
                qml.RZ(x[qubit_idx], wires=qubit_idx)
            elif action[qubit_idx] == 4:
                qml.CNOT(wires=[qubit_idx, (qubit_idx + 1) % len(x)])