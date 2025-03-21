import torch
from torch import nn

from agent import QASEnv
from data import new_data
from trace_distance import compute_trace_distance


class PolicyNetwork(nn.Module):
    def __init__(self, state_sz, action_sz, num_of_qubit):
        super(PolicyNetwork, self).__init__()
        self.action_sz = action_sz
        self.state_linear_relu_stack = nn.Sequential(
            nn.Linear(state_sz * 4, state_sz * 8),
            nn.ReLU(),
            nn.Linear(state_sz * 8, state_sz * 4),
        )
        # qubit 별로 다른 model 적용하기
        self.action_select = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(state_sz * 4, action_sz * 2),
                nn.ReLU(),
                nn.Linear(action_sz * 2, action_sz),
            ) for _ in range(num_of_qubit)]
        )

    def forward(self, state):
        state_new = self.state_linear_relu_stack(state)

        action_probs = []
        epsilon = 0.03

        for qubit_action_select in self.action_select:
            action_prob = torch.softmax(qubit_action_select(state_new), dim=-1)
            adjust_action_probs = (action_prob + epsilon) / (
                    1 + epsilon * self.action_sz)
            action_probs.append(adjust_action_probs)

        return torch.stack(action_probs, dim=0)


class PolicyNetworkComplex(nn.Module):
    def __init__(self, state_sz, action_sz, num_of_qubit):
        super(PolicyNetworkComplex, self).__init__()
        self.action_sz = action_sz
        self.action_select = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(state_sz * 4, state_sz * 8),
                nn.ReLU(),
                nn.Linear(state_sz * 8, state_sz * 16),
                nn.ReLU(),
                nn.Linear(state_sz * 16, action_sz * 16),
                nn.ReLU(),
                nn.Linear(action_sz * 16, action_sz * 4),
                nn.ReLU(),
                nn.Linear(action_sz * 4, action_sz),
            ) for _ in range(num_of_qubit)]
        )

    def forward(self, state):
        action_probs = []
        epsilon = 0.03

        for qubit_action_select in self.action_select:
            action_prob = torch.softmax(qubit_action_select(state), dim=-1)
            adjust_action_probs = (action_prob + epsilon) / (
                    1 + epsilon * self.action_sz)
            action_probs.append(adjust_action_probs)

        return torch.stack(action_probs, dim=0)


def train_policy(X_train_transformed, batch_sz, data_sz, Y_train, policy,
                 optimizer, env, episodes, gamma, trace_dist, max_steps,
                 NQE_model):
    policy_losses = []
    trace_dists = []

    # Prepare samples from each class, 그런데 이거를 transformed로 하는게 맞나?
    X_class0 = [x for x, y in zip(X_train_transformed, Y_train) if y == 0]
    X_class1 = [x for x, y in zip(X_train_transformed, Y_train) if y == 1]

    # Limit the number of samples to keep computation manageable
    X_class0 = X_class0[:25]
    X_class1 = X_class1[:25]

    for episode in range(episodes):
        X1_batch, X2_batch, Y_batch = new_data(batch_sz, X_train_transformed,
                                               Y_train)
        state, _ = env.reset()
        done = False
        log_probs = []
        rewards = []
        prev_action = torch.tensor([999 for _ in range(data_sz)])

        while not done:
            prob = policy.forward(state)
            dist = torch.distributions.Categorical(prob)
            action = dist.sample()
            mask = (action == prev_action)
            while mask.any():
                new_samples = dist.sample()
                action[mask] = new_samples[mask]
                mask = (action == prev_action)
            prev_action = action

            next_state, reward, done = env.step(action=action,
                                                X1=X1_batch.numpy(),
                                                X2=X2_batch.numpy(),
                                                Y_batch=Y_batch)
            log_prob = dist.log_prob(action.clone().detach())
            log_probs.append(log_prob.sum())
            rewards.append(reward)
            state = next_state

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        log_probs = torch.stack(log_probs)
        policy_loss = -log_probs * returns
        policy_loss = policy_loss.mean()
        policy_losses.append(policy_loss)

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        print(f'Episode {episode + 1}/{episodes}, Loss: {policy_loss.item()}')

        if trace_dist and episode % 10 == 0:
            # Generate the current action_sequence
            action_seq = generate_policy_action_sequence(policy_model=policy,
                                                         batch_sz=batch_sz,
                                                         data_sz=data_sz,
                                                         X_train_transformed=X_train_transformed,
                                                         max_steps=max_steps)
            # Compute the trace distance
            trace_dist = compute_trace_distance(data_sz=data_sz,
                                                NQE_model=NQE_model,
                                                action_seq=action_seq,
                                                X_class0=X_class0,
                                                X_class1=X_class1)
            trace_dists.append(trace_dist)
            print(f'Episode {episode + 1}, Trace Distance: {trace_dist}')

    policy_losses = [loss.item() for loss in policy_losses]

    if trace_dist:
        return policy, policy_losses, trace_dists
    else:
        return policy, policy_losses


# Function to generate action sequence
def generate_policy_action_sequence(policy_model, batch_sz, data_sz,
                                    X_train_transformed, max_steps):
    env_eval = QASEnv(num_of_qubit=data_sz,
                      max_timesteps=max_steps,
                      batch_sz=batch_sz)
    state, _ = env_eval.reset()
    action_seq = []
    prev_action = torch.tensor(
        [999 for _ in range(env_eval.simulator.num_wires)])

    for i in range(max_steps):
        with torch.no_grad():
            prob = policy_model(state)
            dist = torch.distributions.Categorical(prob)
            action = dist.sample()
            mask = (action == prev_action)
            while mask.any():
                new_samples = dist.sample()
                action[mask] = new_samples[mask]
                mask = (action == prev_action)
            prev_action = action
        action_seq.append(action.numpy())
        state = env_eval.step_eval(action, X_train_transformed)
        if i % 3 == 0:
            print(f'{i}/{max_steps} actions generated')

    return action_seq
