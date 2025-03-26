from datetime import datetime

import torch

from REINFORCE.analysis import fidelity_plot, plot_circuit
from REINFORCE.data import data_load_and_process, new_data
from REINFORCE.fidelity import check_fidelity
from REINFORCE.initializer import initialize_circuit
from REINFORCE.modifier import remover, inserter
from REINFORCE.policy import PolicyInsertWithMask, PolicyRemove, insert_gate_map
from REINFORCE.utils import fill_identity_gates, representer, FixedLinearProjection, sample_remove_position, \
    sample_insert_gate_param, ordering, check_circuit_structure

if __name__ == "__main__":
    print(datetime.now())

    num_of_qubit = 4
    gate_types = ["RX", "RY", "RZ", "CNOT", "H"]
    depth = 25

    representation_dim = 64
    policy_dim = 64
    batch_size = 25
    gamma = 0.99
    learning_rate = 0.001
    max_episode = 300
    max_step = 100
    fidelity_drop_threshold = 0.5

    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='kmnist', reduction_sz=num_of_qubit)
    circuit_original = initialize_circuit("random", depth, num_of_qubit, gate_types)  # either 'zz' or 'random'
    circuit_original = fill_identity_gates(circuit_original, num_of_qubit, depth)

    tensor = representer(circuit_info=circuit_original, num_qubits=num_of_qubit, depth=depth, gate_types=gate_types)
    projection = FixedLinearProjection(in_dim=tensor.shape[1], out_dim=representation_dim)

    policy_remove = PolicyRemove(input_dim=representation_dim, hidden_dim=policy_dim)
    policy_insert = PolicyInsertWithMask(input_dim=representation_dim, hidden_dim=policy_dim)

    policy_remove.train()
    policy_insert.train()

    opt_remove = torch.optim.Adam(policy_remove.parameters(), lr=learning_rate)
    opt_insert = torch.optim.Adam(policy_insert.parameters(), lr=learning_rate)

    fidelity_logs = []

    for episode in range(max_episode):
        circuit = circuit_original.copy()
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)

        low_fidelity_steps = 0
        initial_fidelity = check_fidelity(circuit, X1_batch, X2_batch, Y_batch)
        fidelity_threshold = initial_fidelity * fidelity_drop_threshold

        log_probs = []
        rewards = []

        for step in range(max_step):
            print(f"step {step}")
            # plot_circuit(circuit, num_of_qubit)
            # print("#########")
            circuit_representation = representer(circuit, num_of_qubit, depth, gate_types)
            dense_representation = projection(circuit_representation)

            remove_prob_tensor = policy_remove(dense_representation)
            qubit_index, depth_index = sample_remove_position(remove_prob_tensor)
            remove_log_prob = torch.log(remove_prob_tensor[0, depth_index, qubit_index] + 1e-8)

            circuit_removed = remover(circuit, qubit_index, depth_index)
            circuit_removed = ordering(circuit_removed)
            circuit_removed_representation = representer(circuit_removed, num_of_qubit, depth, gate_types)
            dense_removed_representation = projection(circuit_removed_representation)

            insert_prob_tensor = policy_insert(dense_removed_representation, qubit_index, depth_index)
            insert_decision = sample_insert_gate_param(insert_prob_tensor, insert_gate_map, qubit_index, num_of_qubit,
                                                       circuit_removed, depth_index)
            insert_log_prob = torch.log(insert_prob_tensor[insert_prob_tensor.argmax()] + 1e-8)

            circuit_inserted = inserter(circuit_removed, depth_index, insert_decision)
            circuit_inserted = ordering(circuit_inserted)
            circuit_inserted = fill_identity_gates(circuit_inserted, num_of_qubit, depth)
            inserted_fidelity = check_fidelity(circuit_inserted, X1_batch, X2_batch, Y_batch)

            reward = -inserted_fidelity
            circuit = circuit_inserted

            log_probs.append(remove_log_prob + insert_log_prob)
            rewards.append(reward)

            if inserted_fidelity <= fidelity_threshold:
                low_fidelity_steps += 1
            else:
                low_fidelity_steps = 0

            if low_fidelity_steps >= 10:
                done = True
                print("done activated")
                break

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G

        opt_remove.zero_grad()
        opt_insert.zero_grad()
        loss.backward()
        opt_remove.step()
        opt_insert.step()

        fidelity_logs.append(inserted_fidelity)
        print(f"[episode {episode}] Fidelity: {inserted_fidelity:.4f}, Reward: {reward:.4f}, Steps: {step + 1}")

    fidelity_plot(fidelity_logs)
