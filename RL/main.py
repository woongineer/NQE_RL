from NQE_RL_iterator import NQE_RL_iterator
from QCNN import circuit_training, QCNN_classifier, accuracy_test
from data import data_load_and_process
from figures import plot_comparison

if __name__ == "__main__":
    # Number of total iterations
    total_iter = 2

    # Parameter settings
    data_sz = 4  # Data reduction size from 256->, determine # of qubit
    batch_sz = 25

    # Parameter for NQE
    NQE_iter = 2

    # Parameter for RL
    gamma = 0.98
    RL_lr = 0.01
    state_sz = data_sz ** 2
    action_sz = 5  # Number of possible actions, RX, RY, RZ, H, CX
    episodes = 220  ## TODO DQN의 경우는 warm up start때문에 2000번쯤 해야하나?
    max_steps = 8

    # Parameter for DQN, TODO Dynamic for now, need to check
    buffer_sz = episodes * (max_steps-1) * 2
    warm_up = int(buffer_sz / 20)
    target_interval = int(buffer_sz / 10)

    # Parameters for QCNN
    QCNN_steps = 2
    QCNN_lr = 0.01
    QCNN_batch_sz = 25

    # RL model to use
    RL_model = 'DQN'  # policy_gradient, policy_gradient_complex, DQN

    # Compute trace_distance
    trace_dist = True

    # Load data
    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='kmnist',
                                                             reduction_sz=data_sz)

    NQE_models, action_seqs = NQE_RL_iterator(RL_model=RL_model,
                                              total_iter=total_iter,
                                              NQE_iter=NQE_iter,
                                              episodes=episodes,
                                              batch_sz=batch_sz,
                                              state_sz=state_sz,
                                              action_sz=action_sz,
                                              data_sz=data_sz,
                                              RL_lr=RL_lr,
                                              max_steps=max_steps,
                                              gamma=gamma,
                                              X_train=X_train,
                                              Y_train=Y_train,
                                              trace_dist=trace_dist,
                                              buffer_sz=buffer_sz,
                                              warm_up=warm_up,
                                              target_interval=target_interval)

    # After iterations, use the final NQE model and action_sequence for QCNN
    first_NQE_model = NQE_models[0]
    last_NQE_model = NQE_models[-1]
    last_action_seq = action_seqs[-1]

    # Convert labels for QCNN
    Y_train_QCNN = [-1 if y == 0 else 1 for y in Y_train]
    Y_test_QCNN = [-1 if y == 0 else 1 for y in Y_test]

    # Train QCNN with final NQE and action_sequence
    loss_history_none, weight_none = circuit_training(
        QCNN_lr=QCNN_lr,
        QCNN_steps=QCNN_steps,
        batch_sz=QCNN_batch_sz,
        X_train=X_train,
        Y_train=Y_train_QCNN,
        scheme=None)
    loss_history_NQE, weight_NQE = circuit_training(
        QCNN_lr=QCNN_lr,
        QCNN_steps=QCNN_steps,
        batch_sz=QCNN_batch_sz,
        X_train=X_train,
        Y_train=Y_train_QCNN,
        scheme="NQE",
        NQE_model=first_NQE_model)
    loss_history_NQE_RL, weight_NQE_RL = circuit_training(
        QCNN_lr=QCNN_lr,
        QCNN_steps=QCNN_steps,
        batch_sz=QCNN_batch_sz,
        X_train=X_train,
        Y_train=Y_train_QCNN,
        scheme="NQE_RL",
        NQE_model=last_NQE_model,
        action_seq=last_action_seq)

    # Evaluate QCNN
    pred_none = [QCNN_classifier(weight_none, x, None) for x in X_test]
    pred_NQE = [QCNN_classifier(weight_NQE, x, "NQE",
                                first_NQE_model) for x in X_test]
    pred_NQE_RL = [QCNN_classifier(weight_NQE_RL, x, "NQE_RL", last_NQE_model,
                                   last_action_seq) for x in X_test]

    accuracy_none = accuracy_test(pred_none, Y_test_QCNN) * 100
    accuracy_NQE = accuracy_test(pred_NQE, Y_test_QCNN) * 100
    accuracy_NQE_RL = accuracy_test(pred_NQE_RL, Y_test_QCNN) * 100

    plot_comparison(RL_model,
                    loss_history_none, loss_history_NQE, loss_history_NQE_RL,
                    accuracy_none, accuracy_NQE, accuracy_NQE_RL)

    print(f"Accuracy without NQE: {accuracy_none:.3f}")
    print(f"Accuracy with NQE: {accuracy_NQE:.3f}")
    print(f"Accuracy with NQE & RL_legacy: {accuracy_NQE_RL:.3f}")
