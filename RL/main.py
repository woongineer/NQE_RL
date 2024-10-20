from NQE_RL_iterator import NQE_RL_iterator
from QCNN import circuit_training, QCNN_classifier, accuracy_test
from data import data_load_and_process
from figures import plot_comparison

if __name__ == "__main__":
    # Number of total iterations
    total_iterations = 2

    # Parameter settings
    data_size = 4  # Data reduction size from 256->, determine # of qubit
    batch_size = 25

    # Parameter for NQE
    NQE_iterations = 2

    # Parameter for RL_legacy
    gamma = 0.98
    RL_learning_rate = 0.01
    state_size = data_size ** 2
    action_size = 5  # Number of possible actions, RX, RY, RZ, H, CX
    episodes = 22
    max_steps = 8

    # Parameters for QCNN
    QCNN_steps = 2
    QCNN_learning_rate = 0.01
    QCNN_batch_size = 25

    # RL model to use
    RL_model = 'DQN'  # policy_gradient, policy_gradient_complex, DQN

    # Compute trace_distance
    trace_distance = True

    # Load data
    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='kmnist',
                                                             reduction_size=data_size)

    NQE_models, action_sequences = NQE_RL_iterator(RL_model=RL_model,
                                                   total_iterations=total_iterations,
                                                   NQE_iterations=NQE_iterations,
                                                   episodes=episodes,
                                                   batch_size=batch_size,
                                                   state_size=state_size,
                                                   action_size=action_size,
                                                   data_size=data_size,
                                                   RL_learning_rate=RL_learning_rate,
                                                   max_steps=max_steps,
                                                   gamma=gamma,
                                                   X_train=X_train,
                                                   Y_train=Y_train,
                                                   trace_distance=trace_distance)

    # After iterations, use the final NQE model and action_sequence for QCNN
    first_NQE_model = NQE_models[0]
    final_NQE_model = NQE_models[-1]
    final_action_sequence = action_sequences[-1]

    # Convert labels for QCNN
    Y_train_QCNN = [-1 if y == 0 else 1 for y in Y_train]
    Y_test_QCNN = [-1 if y == 0 else 1 for y in Y_test]

    # Train QCNN with final NQE and action_sequence
    loss_history_with_none, weight_with_none = circuit_training(
        QCNN_learning_rate=QCNN_learning_rate,
        QCNN_steps=QCNN_steps,
        batch_size=QCNN_batch_size,
        X_train=X_train,
        Y_train=Y_train_QCNN,
        scheme=None)
    loss_history_with_NQE, weight_with_NQE = circuit_training(
        QCNN_learning_rate=QCNN_learning_rate,
        QCNN_steps=QCNN_steps,
        batch_size=QCNN_batch_size,
        X_train=X_train,
        Y_train=Y_train_QCNN,
        scheme="NQE",
        NQE_model=first_NQE_model)
    loss_history_with_NQE_RL, weight_with_NQE_RL = circuit_training(
        QCNN_learning_rate=QCNN_learning_rate,
        QCNN_steps=QCNN_steps,
        batch_size=QCNN_batch_size,
        X_train=X_train,
        Y_train=Y_train_QCNN,
        scheme="NQE_RL",
        NQE_model=final_NQE_model,
        action_sequence=final_action_sequence)

    # Evaluate QCNN
    prediction_with_none = [QCNN_classifier(weight_with_none, x, None) for x in
                            X_test]
    prediction_with_NQE = [
        QCNN_classifier(weight_with_NQE, x, "NQE", first_NQE_model) for x in
        X_test]
    prediction_with_NQE_RL = [
        QCNN_classifier(weight_with_NQE_RL, x, "NQE_RL", final_NQE_model,
                        final_action_sequence) for x in X_test]

    accuracy_with_none = accuracy_test(prediction_with_none, Y_test_QCNN) * 100
    accuracy_with_NQE = accuracy_test(prediction_with_NQE, Y_test_QCNN) * 100
    accuracy_with_NQE_RL = accuracy_test(prediction_with_NQE_RL,
                                         Y_test_QCNN) * 100

    plot_comparison(RL_model, loss_history_with_none, loss_history_with_NQE,
                    loss_history_with_NQE_RL,
                    accuracy_with_none, accuracy_with_NQE, accuracy_with_NQE_RL)

    print(f"Accuracy without NQE: {accuracy_with_none:.3f}")
    print(f"Accuracy with NQE: {accuracy_with_NQE:.3f}")
    print(f"Accuracy with NQE & RL_legacy: {accuracy_with_NQE_RL:.3f}")
