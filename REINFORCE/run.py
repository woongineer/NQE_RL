from datetime import datetime

import torch
import numpy as np

from REINFORCE.data import data_load_and_process, new_data
from REINFORCE.initializer import initialize_circuit
from REINFORCE.utils import fill_identity_gates, plot_circuit, representer, represent_circuit_tensor, \
    FixedLinearProjection
from REINFORCE.policy import PolicyInsert, PolicyRemove
from REINFORCE.fidelity import check_fidelity


# from REINFORCE.utils import circuit_profiler, check_fidelity, representer,
# from REINFORCE.modifier.py import inserter, remover


if __name__ == "__main__":
    print(datetime.now())

    num_of_qubit = 4
    gate_types = ["RX", "RY", "RZ", "CNOT", "H"]
    depth = 8

    representation_dim = 64
    policy_dim = 64
    batch_size = 25
    gamma = 0.99
    learning_rate = 0.001
    max_epoch = 300
    max_step = 100
    done = 0.8  # 진전이 없을때 done을 하거나 원래 fidelity의 80%로 떨어지면??

    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='kmnist', reduction_sz=num_of_qubit)
    circuit = initialize_circuit("random", depth, num_of_qubit, gate_types)  # either 'zz' or 'random'
    filled_circuit = fill_identity_gates(circuit, num_of_qubit, depth)

    plot_circuit(circuit, num_of_qubit)  # circuit/filled_circuit

    circuit_representation = representer(circuit, num_of_qubit, depth, gate_types)

    tensor = represent_circuit_tensor(circuit_info=circuit,
                                      num_qubits=num_of_qubit,
                                      depth=depth,
                                      gate_types=gate_types)

    projection = FixedLinearProjection(in_dim=tensor.shape[1], out_dim=representation_dim)
    dense_tensor = projection(tensor)

    policy_remove = PolicyRemove(input_dim=representation_dim, hidden_dim=policy_dim)
    policy_insert = PolicyInsert(input_dim=representation_dim, hidden_dim=policy_dim)

    policy_remove.train()
    policy_insert.train()

    opt_remove = torch.optim.Adam(policy_remove.parameters(), lr=learning_rate)
    opt_insert = torch.optim.Adam(policy_insert.parameters(), lr=learning_rate)


    for epoch in range(max_epoch):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        initial_fidelity = check_fidelity(circuit, X1_batch, X2_batch, Y_batch)
        done = False
        log_probs = []
        rewards = []

        while not done:
            circuit_representation = representer(circuit, num_of_qubit, depth, gate_types)
            qubit_index, depth_index = policy_remove(circuit_representation)

            circuit_removed = remover(circuit, (qubit_index, depth_index))
            circuit_removed_representation = representer(circuit_removed, data_size, depth, gate_types, inputs)

            gate_type, parameter_index = policy_insert(circuit_removed_representation, (qubit_index, depth_index))
            circuit_inserted = inserter(circuit_removed, (qubit_index, depth_index), (gate_type, parameter_index))
            inserted_fidelity = check_fidelity(circuit_inserted)

            reward = gamma * (inserted_fidelity - initial_fidelity)
            if reward > done:
                break
