import random

import numpy as np
import pennylane as qml
import plotly.graph_objects as go
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde


import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# 함수 정의
def plot_distributions(data, output_filename):
    fig = go.Figure()

    # 파란색 계열 (key = 0)
    blue_colors = [f"rgba(0, 0, {int(255 - i * 3.5)}, 0.6)" for i in range(64)]
    # 빨간색 계열 (key = 1)
    red_colors = [f"rgba({int(255 - i * 3.5)}, 0, 0, 0.6)" for i in range(64)]
    # 초록색 계열 (key = 2)
    green_colors = [f"rgba(0, {int(255 - i * 3.5)}, 0, 0.6)" for i in range(64)]

    # 데이터 플롯
    for main_key in data:
        if main_key == 0:
            color_list = blue_colors
        elif main_key == 1:
            color_list = red_colors
        elif main_key == 2:
            color_list = green_colors
        else:
            continue

        for sub_key, values in data[main_key].items():
            # KDE 계산
            kde = gaussian_kde(values)
            x_range = np.linspace(min(values), max(values), 500)
            density = kde(x_range)

            # 이름 설정
            trace_name = f"{main_key}-{sub_key}"

            # 분포 추가
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=density,
                    mode='lines',
                    line=dict(color=color_list[sub_key]),
                    name=trace_name
                )
            )

    # 레이아웃 설정
    fig.update_layout(
        title="Gaussian KDE Distributions",
        xaxis_title="Value",
        yaxis_title="Density",
        template="plotly_white",
        legend_title="Distributions"
    )

    # HTML로 저장
    fig.write_html(output_filename)
    print(f"Plot saved to {output_filename}")


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


def quantum_embedding(x, gate_structure):
    for gate, qubit_idx, data_index in gate_structure:
        if gate == 'RX':
            qml.RX(x[data_index], wires=qubit_idx)
        elif gate == 'RY':
            qml.RY(x[data_index], wires=qubit_idx)
        elif gate == 'RZ':
            qml.RZ(x[data_index], wires=qubit_idx)
        elif gate == 'CNOT':
            qml.CNOT(wires=[qubit_idx[0], qubit_idx[1]])

class NQEModel(nn.Module):
    def __init__(self, dev, gate_structure):
        super().__init__()

        @qml.qnode(dev, interface='torch')
        def circuit(inputs):
            quantum_embedding(inputs[0:4], gate_structure)
            qml.adjoint(quantum_embedding)(inputs[4:8], gate_structure)

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