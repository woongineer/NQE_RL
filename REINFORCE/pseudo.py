circuit = initialize_circuit("zz", 3)
(number_of_qubit, number_of_depth) = circuit_profiler(circuit)
policy_remove = PolicyRemove(number_of_qubit, number_of_depth)
policy_insert = PolicyInsert(number_of_gate_types, number_of_feature)

for _ in range(max_step):
    initial_fidelity = check_fidelity(circuit)
    circuit_representation = representer(circuit)

    (qubit_index, depth_index) = policy_remove(circuit_representation)
    circuit_removed = remover(circuit, (qubit_index, depth_index))
    circuit_removed_represenation = representer(circuit_removed)
    removed_fidelity = check_fidelity(circuit_removed)

    (gate_type, parameter_index) = policy_insert(circuit_removed_represenation, (qubit_index, depth_index))
    circuit_inserted = inserter(circuit_removed, (qubit_index, depth_index), (gate_type, parameter_index))
    inserted_fidelity = check_fidelity(circuit_inserted)

    """
    initial은 random(depth) or zz(layer 수)
    representation은 graph/cnn/seq
    reward는 fidelity/margin loss
    policy는 nn/rnn/gnn
    action은 gate N개 바꾸기
    """

circuit = initialize_circuit("random", 15, data_size, gate_types)  # either 'zz' or 'random'
for it in range(iterations):
    X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
    fidelity_loss_list = get_fidelity(circuit, X1_batch, X2_batch, Y_batch)

    removal_depth_index, removal_qubit_index, insert_gate_type, insert_other_data_parameter = policy(circuit)
    new_circuit = remove_and_insert_circuit(circuit)

    fidelity_loss_list_new = get_fidelity(new_circuit, X1_batch, X2_batch, Y_batch)

    reward = fidelity_loss_list - fidelity_loss_list_new
    update using reward

    circuit = new_circuit