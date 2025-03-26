import matplotlib.pyplot as plt
import pennylane as qml


def fidelity_plot(fidelity_logs):
    plt.figure(figsize=(10, 4))
    plt.plot(fidelity_logs, marker='o')
    plt.title("Fidelity Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Fidelity Loss (1 - Fidelity)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fidelity_loss_over_epochs.png")
    plt.show()


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
