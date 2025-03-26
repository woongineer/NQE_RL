import pennylane as qml
import torch

dev = qml.device('default.qubit', wires=4)


def apply_gate(gate, x):
    gate_type = gate['gate_type']
    param_idx = gate['param']
    param_value = x[param_idx]

    ###########수정된 부분##########
    # qubit index 정의
    qubits = gate['qubits']
    if gate_type == 'CNOT':
        control, target = qubits
    else:
        target = qubits[0]
    ###########수정된 부분##########

    # gate 적용
    if gate_type == 'RX':
        qml.RX(param_value, wires=target)
    elif gate_type == 'RY':
        qml.RY(param_value, wires=target)
    elif gate_type == 'RZ':
        qml.RZ(param_value, wires=target)
    elif gate_type == 'H':
        qml.Hadamard(wires=target)
    elif gate_type == 'CNOT':
        qml.CNOT(wires=[control, target])
    elif gate_type == 'I':
        pass


def apply_circuit(x, circuit):
    for gate in sorted(circuit, key=lambda g: g['depth']):
        apply_gate(gate, x)


@qml.qnode(dev, interface='torch')
def fidelity_circuit(x1, x2, circuit):
    apply_circuit(x1, circuit)
    qml.adjoint(apply_circuit)(x2, circuit)
    return qml.probs(wires=range(4))


def check_fidelity(circuit, X1_batch, X2_batch, Y_batch):
    preds = []
    for x1, x2 in zip(X1_batch, X2_batch):
        probs = fidelity_circuit(x1, x2, circuit)
        preds.append(probs[0])  # ⟨ψ₁|ψ₂⟩²

    preds = torch.stack(preds)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(preds, Y_batch)
    return loss
