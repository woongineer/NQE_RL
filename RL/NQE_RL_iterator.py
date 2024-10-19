import torch

from DQN import DQNNetwork, train_dqn, ReplayBuffer, \
    generate_DQN_action_sequence
from NQE import train_NQE, transform_data
from agent import QASEnv
from figures import plot_policy_loss, plot_nqe_loss, draw_circuit
from policy_gradient import train_policy, PolicyNetwork, PolicyNetworkComplex, \
    generate_policy_action_sequence


def NQE_RL_iterator(RL_model, total_iterations, NQE_iterations, episodes,
                    batch_size, state_size, action_size, data_size,
                    RL_learning_rate, max_steps, gamma, X_train,
                    Y_train, trace_distance):
    if RL_model in ['policy_gradient', 'policy_gradient_complex']:
        return policy_gradient_iterator(RL_model, total_iterations,
                                        NQE_iterations,
                                        episodes,
                                        batch_size, state_size, action_size,
                                        data_size,
                                        RL_learning_rate, max_steps, gamma,
                                        X_train,
                                        Y_train,
                                        trace_distance)
    elif RL_model == 'DQN':
        return DQN_iterator(RL_model, total_iterations,
                            NQE_iterations,
                            episodes,
                            batch_size, state_size, action_size,
                            data_size,
                            RL_learning_rate, max_steps, gamma,
                            X_train,
                            Y_train,
                            trace_distance)


def policy_gradient_iterator(RL_model, total_iterations, NQE_iterations,
                             episodes, batch_size, state_size, action_size,
                             data_size, RL_learning_rate, max_steps, gamma,
                             X_train, Y_train, trace_distance):
    NQE_models = []
    Policy_models = []
    action_sequences = []

    for iter in range(total_iterations):
        print(f"Starting iteration {iter + 1}/{total_iterations}")
        # Step 1: Train NQE
        if iter == 0:
            action_sequence = None
        NQE_model, NQE_losses = train_NQE(X_train, Y_train, NQE_iterations,
                                          batch_size, action_sequence)
        NQE_models.append(NQE_model)

        # Step 2: Transform X_train using NQE_model
        X_train_transformed = transform_data(NQE_model, X_train)

        # Step 3: Train RL_legacy policy
        # PolicyNetwork, PolicyNetworkComplex
        if RL_model == 'policy_gradient':
            policy = PolicyNetwork(state_size=state_size,
                                   action_size=action_size,
                                   num_of_qubit=data_size)
        elif RL_model == 'policy_gradient_complex':
            policy = PolicyNetworkComplex(state_size=state_size,
                                          action_size=action_size,
                                          num_of_qubit=data_size)
        else:
            raise ValueError

        optimizer = torch.optim.Adam(policy.parameters(), lr=RL_learning_rate)
        env = QASEnv(num_of_qubit=data_size, max_timesteps=max_steps,
                     batch_size=batch_size)

        if trace_distance:
            policy, policy_losses, trace_distances = train_policy(
                X_train_transformed, batch_size, data_size, Y_train, policy,
                optimizer, env, episodes,
                gamma, True, max_steps, NQE_model)

        if not trace_distance:
            policy, policy_losses = train_policy(X_train_transformed,
                                                 batch_size,
                                                 data_size, Y_train, policy,
                                                 optimizer, env, episodes,
                                                 gamma, False, max_steps,
                                                 NQE_model)
        Policy_models.append(policy)

        # Step 4: Generate action_sequence
        action_sequence = generate_policy_action_sequence(policy, batch_size,
                                                          data_size,
                                                          X_train_transformed,
                                                          max_steps)
        action_sequences.append(action_sequence)

        # save loss history fig
        plot_nqe_loss(RL_model, NQE_losses, iter)
        plot_policy_loss(RL_model, policy_losses, iter)
        draw_circuit(RL_model, action_sequence, iter)

    return NQE_models, action_sequences


def DQN_iterator(RL_model, total_iterations,
                 NQE_iterations,
                 episodes,
                 batch_size, state_size, action_size,
                 data_size,
                 RL_learning_rate, max_steps, gamma,
                 X_train,
                 Y_train,
                 trace_distance):  ##TODO trace_distance for DQN
    NQE_models = []
    Policy_models = []
    action_sequences = []

    for iter in range(total_iterations):
        print(f"Starting iteration {iter + 1}/{total_iterations}")
        # Step 1: Train NQE
        if iter == 0:
            action_sequence = None
        NQE_model, NQE_losses = train_NQE(X_train, Y_train, NQE_iterations,
                                          batch_size, action_sequence)
        NQE_models.append(NQE_model)

        # Step 2: Transform X_train using NQE_model
        X_train_transformed = transform_data(NQE_model, X_train)

        # Step 3: Train RL_legacy policy
        policy_net = DQNNetwork(state_size=state_size, action_size=action_size,
                                num_of_qubits=data_size)
        target_net = DQNNetwork(state_size=state_size, action_size=action_size,
                                num_of_qubits=data_size)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = torch.optim.Adam(policy_net.parameters(),
                                     lr=RL_learning_rate)
        replay_buffer = ReplayBuffer(capacity=10000)

        target_update = 10

        env = QASEnv(num_of_qubit=data_size, max_timesteps=max_steps,
                     batch_size=batch_size)
        policy_net, total_reward = train_dqn(
            X_train_transformed=X_train_transformed,
            data_size=data_size,
            action_size=action_size,
            Y_train=Y_train,
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            env=env,
            num_episodes=episodes,
            gamma=gamma,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            target_update=target_update
        )
        Policy_models.append(policy_net)

        # Step 4: Generate action_sequence
        action_sequence = generate_DQN_action_sequence(policy_net, batch_size,
                                                       data_size,
                                                       X_train_transformed,
                                                       max_steps)
        action_sequences.append(action_sequence)

        # save loss history fig
        plot_nqe_loss(RL_model, NQE_losses, iter)
        plot_policy_loss(RL_model, total_reward, iter)
        draw_circuit(RL_model, action_sequence, iter)

    return NQE_models, action_sequences
