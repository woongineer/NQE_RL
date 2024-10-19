import numpy as np
import pennylane as qml
import torch


def compute_trace_distance(data_size, NQE_model, action_sequence, X_class0,
                           X_class1):
    dev = qml.device('default.qubit', wires=data_size)

    # Define the quantum circuit using the action_sequence
    def quantum_embedding_rl(x):
        for action in action_sequence:
            for qubit_idx in range(data_size):
                if action[qubit_idx] == 0:
                    qml.Hadamard(wires=qubit_idx)
                elif action[qubit_idx] == 1:
                    qml.RX(x[qubit_idx], wires=qubit_idx)
                elif action[qubit_idx] == 2:
                    qml.RY(x[qubit_idx], wires=qubit_idx)
                elif action[qubit_idx] == 3:
                    qml.RZ(x[qubit_idx], wires=qubit_idx)
                elif action[qubit_idx] == 4:
                    qml.CNOT(wires=[qubit_idx, (qubit_idx + 1) % data_size])

    # Define a qnode to compute the density matrix
    @qml.qnode(dev)
    def density_matrix_circuit(x):
        quantum_embedding_rl(x)
        return qml.density_matrix(wires=range(data_size))

    # Function to get transformed x using NQE_model
    def get_transformed_x(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        x_transformed = NQE_model.linear_relu_stack1(x_tensor)
        x_transformed = x_transformed.squeeze(0).detach().numpy()
        return x_transformed

    # Compute density matrices for samples from each class
    density_matrices_class0 = []
    for x in X_class0:
        x_transformed = get_transformed_x(x)
        rho = density_matrix_circuit(x_transformed)
        density_matrices_class0.append(rho)
    density_matrices_class1 = []
    for x in X_class1:
        x_transformed = get_transformed_x(x)
        rho = density_matrix_circuit(x_transformed)
        density_matrices_class1.append(rho)

    # Average density matrices over samples in each class
    rho0 = np.mean(density_matrices_class0, axis=0)
    rho1 = np.mean(density_matrices_class1, axis=0)

    # Compute trace distance
    rho_diff = rho0 - rho1
    eigvals = np.linalg.eigvals(rho_diff)
    trace_distance = 0.5 * np.sum(np.abs(eigvals))

    return trace_distance
