import pennylane as qml
import numpy as np
import torch
from torch import nn


class FixedLinearProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.linear.weight.requires_grad = False

        # 초기화 방법: 정규분포, 아이덴티티 일부, 등등
        with torch.no_grad():
            torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)

    def forward(self, x):
        # x: (B, C, D, Q) → (B, D, Q, C)
        x = x.permute(0, 2, 3, 1)
        out = self.linear(x)  # (B, D, Q, hidden_dim)
        return out



def project_circuit_tensor(tensor, gate_channel_mapping, embedding_dim=32):
    """
    tensor: torch.Tensor, shape (1, C, D, Q)
    gate_channel_mapping: Dict[str, int], from represent_circuit_tensor()
    embedding_dim: output dim for dense embedding
    return: dense_tensor of shape (1, D, Q, embedding_dim)
    """
    # (1, C, D, Q) → (C, D, Q)
    tensor = tensor.squeeze(0)  # (C, D, Q)
    C, D, Q = tensor.shape

    # Generate projection matrix with meaningfully constructed vectors
    embedding_matrix = generate_gate_projection(gate_channel_mapping, embedding_dim=embedding_dim)  # (C, E)

    # Apply projection: einsum over channel dimension
    # (C, D, Q) x (C, E) → (D, Q, E)
    dense_tensor = torch.einsum('cdq,ce->dqe', tensor, embedding_matrix)

    # Add batch dimension
    dense_tensor = dense_tensor.unsqueeze(0)  # (1, D, Q, E)
    return dense_tensor


def generate_gate_projection(gate_channel_mapping, embedding_dim=32):
    """
    gate_channel_mapping: Dict[str, int]
    embedding_dim: int, output projection dim
    return: torch.Tensor of shape (num_channels, embedding_dim)
    """

    num_channels = len(gate_channel_mapping)
    embedding_matrix = np.zeros((num_channels, embedding_dim), dtype=np.float32)

    for gate_key, idx in gate_channel_mapping.items():
        vec = np.zeros(embedding_dim)

        if gate_key.startswith("RX_"):
            param = int(gate_key.split("_")[1])
            vec[:4] = [1, 0, 0, param / 3]  # RX axis + param normalized

        elif gate_key.startswith("RY_"):
            param = int(gate_key.split("_")[1])
            vec[:4] = [0, 1, 0, param / 3]  # RY axis + param normalized

        elif gate_key.startswith("RZ_"):
            param = int(gate_key.split("_")[1])
            vec[:4] = [0, 0, 1, param / 3]  # RZ axis + param normalized

        elif gate_key == "H":
            vec[:4] = [0.5, 0.5, 0.5, 1.0]  # equal superposition

        elif gate_key.startswith("CNOT_"):
            parts = gate_key.split("_")
            control = int(parts[1])
            target = int(parts[2])
            # simple encoding: control-target diff
            vec[4:8] = np.eye(4)[control] + np.eye(4)[target]  # control/target index embedding

        elif gate_key == "I":
            vec[:4] = [-1, -1, -1, 0]  # 완전 반대 방향

        # normalize vector
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        embedding_matrix[idx] = vec

    return torch.tensor(embedding_matrix, dtype=torch.float32)



def represent_circuit_tensor(circuit_info, num_qubits, depth, gate_types, rotation_set=[0,1,2,3]):
    gate_channel_mapping = {}
    channel_counter = 0

    # 1-qubit rotation gates (RX, RY, RZ) with param index
    for g in gate_types:
        if g.startswith("R"):
            for p in rotation_set:
                gate_channel_mapping[f"{g}_{p}"] = channel_counter
                channel_counter += 1

    # H gate
    if "H" in gate_types:
        gate_channel_mapping["H"] = channel_counter
        channel_counter += 1

    # CNOTs (control ≠ target)
    for q0 in range(num_qubits):
        for q1 in range(num_qubits):
            if q0 == q1:
                continue
            gate_channel_mapping[f"CNOT_{q0}_{q1}"] = channel_counter
            channel_counter += 1

    # 마지막 채널: NoGate (Identity)
    gate_channel_mapping["I"] = channel_counter
    total_channels = channel_counter + 1

    tensor = np.zeros((depth, num_qubits, total_channels), dtype=np.float32)

    for gate in circuit_info:
        d = gate["depth"]
        q0, q1 = gate["qubits"]
        g_type = gate["gate_type"]
        p_idx = int(gate["param"])

        key = None
        if g_type.startswith("R"):
            key = f"{g_type}_{p_idx}"
            if key in gate_channel_mapping:
                ch = gate_channel_mapping[key]
                tensor[d, q0, ch] = 1.0

        elif g_type == "H":
            key = "H"
            ch = gate_channel_mapping[key]
            tensor[d, q0, ch] = 1.0

        elif g_type == "CNOT":
            key = f"CNOT_{q0}_{q1}"
            if key in gate_channel_mapping:
                ch = gate_channel_mapping[key]
                tensor[d, q0, ch] = 1.0
                tensor[d, q1, ch] = 1.0

    # 나머지 채널에 NoGate 표시
    for d in range(depth):
        for q in range(num_qubits):
            if tensor[d, q, :total_channels-1].sum() == 0:
                tensor[d, q, gate_channel_mapping["I"]] = 1.0

    # reshape to (1, C, D, Q)
    tensor = tensor.transpose(2, 0, 1)  # (C, D, Q)
    tensor = tensor[np.newaxis, :, :, :]  # (1, C, D, Q)
    return torch.tensor(tensor, dtype=torch.float32)


def representer(circuit_info, num_qubits, depth, gate_types):
    num_gate_types = len(gate_types)

    gate_type_to_idx = {g: i for i, g in enumerate(gate_types)}

    # shape: (depth, num_qubits, num_gate_types * num_rot)
    tensor = np.zeros((depth, num_qubits, num_gate_types * num_qubits), dtype=np.float32)

    for gate in circuit_info:
        d = gate["depth"]
        q0, q1 = gate["qubits"]
        g_type = gate["gate_type"]
        param = gate["param"]

        gate_idx = gate_type_to_idx[g_type]
        channel_idx = gate_idx * num_qubits + param

        if g_type == "CNOT":
            tensor[d, q0, channel_idx] = 1.0
            tensor[d, q1, channel_idx] = 1.0
        else:
            tensor[d, q0, channel_idx] = 1.0

    # reshape to (1, C, D, Q)
    tensor = tensor.transpose(2, 0, 1)           # (C, D, Q)
    tensor = tensor[np.newaxis, :, :, :]         # (1, C, D, Q)
    return tensor




def fill_identity_gates(circuit_info, num_of_qubit, total_depth):
    filled_circuit_info = []

    for d in range(total_depth):
        # 해당 depth에서 이미 사용된 qubit 찾기
        used_qubits = set()
        for info in circuit_info:
            if info["depth"] == d:
                used_qubits.add(info["qubits"][0])
                if info["gate_type"] == "CNOT":
                    used_qubits.add(info["qubits"][1])
                filled_circuit_info.append(info)

        # 사용되지 않은 qubit에는 Identity gate 추가
        for q in range(num_of_qubit):
            if q not in used_qubits:
                filled_circuit_info.append({
                    "gate_type": "I",
                    "depth": d,
                    "qubits": (q, q),
                    "param": None
                })

    # depth 순으로 정렬해서 리턴
    filled_circuit_info.sort(key=lambda x: x["depth"])
    return filled_circuit_info



def plot_circuit(circuit_info, num_of_qubit):
    dev = qml.device('default.qubit', wires=num_of_qubit)

    @qml.qnode(dev)
    def quantum_circuit():
        for gate in circuit_info:
            gate_type = gate["gate_type"]
            qubits = gate["qubits"]
            param = gate["param"]

            if gate_type == "RX":
                qml.RX(param, wires=qubits[0])
            elif gate_type == "RY":
                qml.RY(param, wires=qubits[0])
            elif gate_type == "RZ":
                qml.RZ(param, wires=qubits[0])
            elif gate_type == "RX_arctan":
                qml.RX(np.arctan(param), wires=qubits[0])
            elif gate_type == "RY_arctan":
                qml.RY(np.arctan(param), wires=qubits[0])
            elif gate_type == "RZ_arctan":
                qml.RZ(np.arctan(param), wires=qubits[0])
            elif gate_type == "CNOT":
                qml.CNOT(wires=qubits)
            elif gate_type == "H":
                qml.Hadamard(wires=qubits[0])
            elif gate_type == "I":
                qml.Identity(wires=qubits[0])
            else:
                raise ValueError(f"Unknown gate type: {gate_type}")

        return qml.expval(qml.PauliZ(0))

    drawer = qml.draw(quantum_circuit)
    print(drawer())

