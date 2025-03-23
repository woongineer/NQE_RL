from datetime import datetime

import torch

from REINFORCE.data import data_load_and_process, new_data
# from REINFORCE.policy import PolicyInsert, PolicyRemove
from REINFORCE.initializer import initialize_circuit
# from REINFORCE.utils import circuit_profiler, check_fidelity, representer
# from REINFORCE.modifier import inserter, remover


if __name__ == "__main__":
    print(datetime.now())

    data_size = 4
    gate_types = ["RX", "RY", "RZ", "CNOT", "H", "I" "RX_arctan", "RY_arctan", "RZ_arctan"]

    gamma = 0.99
    learning_rate = 0.001
    max_epoch = 300
    max_step = 20
    done = 0.8

    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='kmnist', reduction_sz=data_size)
    circuit = initialize_circuit([0, 0, 0, 0], "random", 15, data_size, gate_types)  # either 'zz' or 'random'

    num_of_qubit, num_of_depth = circuit_profiler(circuit)
    policy_remove = PolicyRemove(num_of_qubit, num_of_depth)
    policy_insert = PolicyInsert(len(gate_types), data_size)

    policy_remove.train()
    policy_insert.train()

    opt_remove = torch.optim.Adam(policy_remove.parameters(), lr=learning_rate)
    opt_insert = torch.optim.Adam(policy_insert.parameters(), lr=learning_rate)

    for epoch in range(max_epoch):
        initial_fidelity = check_fidelity(circuit)

        for step in range(max_step):
            circuit_representation = representer(circuit)

            qubit_index, depth_index = policy_remove(circuit_representation)
            circuit_removed = remover(circuit, (qubit_index, depth_index))
            circuit_removed_representation = representer(circuit_removed)
            removed_fidelity = check_fidelity(circuit_removed)

            gate_type, parameter_index = policy_insert(circuit_removed_representation, (qubit_index, depth_index))
            circuit_inserted = inserter(circuit_removed, (qubit_index, depth_index), (gate_type, parameter_index))
            inserted_fidelity = check_fidelity(circuit_inserted)

            reward = gamma * (inserted_fidelity - initial_fidelity)
            if reward > done:
                break








