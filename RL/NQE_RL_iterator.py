import torch

from DQN import DQNNetwork, train_dqn, ReplayBuffer, \
    generate_DQN_action_sequence
from NQE import train_NQE, transform_data
from agent import QASEnv
from figures import plot_policy_loss, plot_nqe_loss, draw_circuit
from policy_gradient import train_policy, PolicyNetwork, PolicyNetworkComplex, \
    generate_policy_action_sequence


def NQE_RL_iterator(RL_model, total_iter, NQE_iter, episodes,
                    batch_sz, state_sz, action_sz, data_sz,
                    RL_lr, max_steps, gamma, X_train,
                    Y_train, trace_dist,
                    buffer_sz=0, warm_up=0, target_interval=0):
    if RL_model in ['policy_gradient', 'policy_gradient_complex']:
        return policy_gradient_iterator(RL_model=RL_model,
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
                                        trace_dist=trace_dist)
    elif RL_model == 'DQN':
        if buffer_sz == 0:
            raise ValueError('Check buffer size, should not be 0')
        if batch_sz > warm_up:
            raise ValueError(
                f'warm_up={warm_up} must be larger than warm_up batch_sz={batch_sz}')
        return DQN_iterator(RL_model=RL_model,
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


def policy_gradient_iterator(RL_model, total_iter, NQE_iter,
                             episodes, batch_sz, state_sz, action_sz,
                             data_sz, RL_lr, max_steps, gamma,
                             X_train, Y_train, trace_dist):
    NQE_models = []
    Policy_models = []
    action_seqs = []

    for iter in range(total_iter):
        print(f"Starting iteration {iter + 1}/{total_iter}")
        # Step 1: Train NQE
        if iter == 0:
            action_seq = None
        NQE_model, NQE_losses = train_NQE(X_train=X_train,
                                          Y_train=Y_train,
                                          NQE_iter=NQE_iter,
                                          batch_sz=batch_sz,
                                          action_seq=action_seq)
        NQE_models.append(NQE_model)

        # Step 2: Transform X_train using NQE_model
        X_train_transformed = transform_data(NQE_model=NQE_model,
                                             X_data=X_train)

        # Step 3: Train RL_legacy policy
        # PolicyNetwork, PolicyNetworkComplex
        if RL_model == 'policy_gradient':
            policy = PolicyNetwork(state_sz=state_sz,
                                   action_sz=action_sz,
                                   num_of_qubit=data_sz)
        elif RL_model == 'policy_gradient_complex':
            policy = PolicyNetworkComplex(state_sz=state_sz,
                                          action_sz=action_sz,
                                          num_of_qubit=data_sz)
        else:
            raise ValueError

        optimizer = torch.optim.Adam(policy.parameters(), lr=RL_lr)
        env = QASEnv(num_of_qubit=data_sz,
                     max_timesteps=max_steps,
                     batch_sz=batch_sz)

        if trace_dist:
            policy, policy_losses, trace_dist = train_policy(
                X_train_transformed=X_train_transformed,
                batch_sz=batch_sz,
                data_sz=data_sz,
                Y_train=Y_train,
                policy=policy,
                optimizer=optimizer,
                env=env,
                episodes=episodes,
                gamma=gamma,
                trace_dist=True,
                max_steps=max_steps,
                NQE_model=NQE_model
            )

        if not trace_dist:
            policy, policy_losses = train_policy(
                X_train_transformed=X_train_transformed,
                batch_sz=batch_sz,
                data_sz=data_sz,
                Y_train=Y_train,
                policy=policy,
                optimizer=optimizer,
                env=env,
                episodes=episodes,
                gamma=gamma,
                trace_dist=False,
                max_steps=max_steps,
                NQE_model=NQE_model
            )

        Policy_models.append(policy)

        # Step 4: Generate action_seq
        action_seq = generate_policy_action_sequence(policy_model=policy,
                                                     batch_sz=batch_sz,
                                                     data_sz=data_sz,
                                                     X_train_transformed=X_train_transformed,
                                                     max_steps=max_steps)
        action_seqs.append(action_seq)

        # save loss history fig
        plot_nqe_loss(RL_model, NQE_losses, iter)
        plot_policy_loss(RL_model, policy_losses, iter)
        draw_circuit(RL_model, action_seq, iter)

    return NQE_models, action_seqs


def DQN_iterator(RL_model, total_iter, NQE_iter, episodes,
                 batch_sz, state_sz, action_sz, data_sz,
                 RL_lr, max_steps, gamma,
                 X_train, Y_train,
                 buffer_sz, warm_up, target_interval,
                 trace_dist):  ##TODO trace_distance for DQN
    NQE_models = []
    Policy_models = []
    action_seqs = []

    for iter in range(total_iter):
        print(f"Starting iteration {iter + 1}/{total_iter}")
        # Step 1: Train NQE
        if iter == 0:
            action_seq = None
        NQE_model, NQE_losses = train_NQE(X_train, Y_train, NQE_iter,
                                          batch_sz, action_seq)
        NQE_models.append(NQE_model)

        # Step 2: Transform X_train using NQE_model
        X_train_transformed = transform_data(NQE_model, X_train)

        # Step 3: Train RL_legacy policy
        q_policy = DQNNetwork(state_sz=state_sz, action_sz=action_sz,
                              num_of_qubits=data_sz)
        q_target = DQNNetwork(state_sz=state_sz, action_sz=action_sz,
                              num_of_qubits=data_sz)
        q_target.load_state_dict(q_policy.state_dict())
        q_target.eval()

        optimizer = torch.optim.Adam(q_policy.parameters(), lr=RL_lr)
        replay_buffer = ReplayBuffer(capacity=buffer_sz)

        env = QASEnv(num_of_qubit=data_sz,
                     max_timesteps=max_steps,
                     batch_sz=batch_sz)
        q_policy, total_reward = train_dqn(
            X_train_transformed=X_train_transformed,
            data_sz=data_sz,
            action_sz=action_sz,
            Y_train=Y_train,
            q_policy=q_policy,
            q_target=q_target,
            optimizer=optimizer,
            env=env,
            num_episodes=episodes,
            gamma=gamma,
            replay_buffer=replay_buffer,
            batch_sz=batch_sz,
            warm_up=warm_up,
            target_interval=target_interval
        )
        Policy_models.append(q_policy)

        # Step 4: Generate action_sequence
        action_seq = generate_DQN_action_sequence(q_policy=q_policy,
                                                  batch_sz=batch_sz,
                                                  data_sz=data_sz,
                                                  X_train_transformed=X_train_transformed,
                                                  max_steps=max_steps)
        action_seqs.append(action_seq)

        # save loss history fig
        plot_nqe_loss(RL_model, NQE_losses, iter)
        plot_policy_loss(RL_model, total_reward, iter)
        draw_circuit(RL_model, action_seq, iter)

    return NQE_models, action_seqs
