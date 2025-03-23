# use code from (all explanations there)
# https://www.kaggle.com/code/minaiyuki/muzero-model-based-rl-with-pytorch

import random
from collections import deque
import math
import numpy as np
import pennylane as qml
import torch
from pennylane import numpy as np
from torch import nn

from data import data_load_and_process as dataprep
from data import new_data
import numpy as np
import collections
from collections import deque
import gym
import itertools
import random
import os
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import softmax

# Set your device
n_qubit = 4
dev = qml.device('default.qubit', wires=n_qubit)


class Node(object):

    def __init__(self, prior):
        """
        Node in MCTS
        prior: The prior policy on the node, computed from policy network
        """
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.latent_state = None
        self.reward = 0
        self.expanded = False

    def value(self):
        """
        Compute expected value of a node
        """
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count


class Game:
    """
    A single episode of interaction with the environment.
    """

    def __init__(self, action_size, discount, current_state):
        self.action_size = action_size
        self.current_state = current_state
        self.done = False
        self.discount = discount

        self.state_history = [self.current_state]
        self.action_history = []
        self.reward_history = []

        self.root_values = []
        self.child_visits = []

    def store_search_stats(self, root):
        """Store the search stats for the current root node
        root: Node object including the infos of the current root node
        """
        # Stores the normalized root node child visits (i.e. policy target)
        sum_visits = sum(child.visit_count for child in root.children.values())

        visits = []
        for action in range(self.action_size):
            if action in root.children:
                visits.append(root.children[action].visit_count / sum_visits)
            else:
                visits.append(0)
        self.child_visits.append(np.array(visits))

        # Stores the root node value, computed from the MCTS (i.e. value target)
        self.root_values.append(root.value())

    def take_action(self, action, env):
        """Take an action and store the action/reward/new_state into history
        """
        observation, reward, done = env.step(action)
        self.current_state = observation
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.done = done
        if not self.done:
            self.state_history.append(self.current_state)

    def make_target(self, state_idx, unrolls, td_steps):
        """Makes the target data for training

        state_idx: the start state
        unrolls: how many times to unroll from the current state each unroll
                 forms a new target
        td_steps: the number of temporal difference steps used in bootstrapping
                  the value function
        """
        targets = []  # target = (value, reward, policy)
        actions = []

        for cur_idx in range(state_idx, state_idx + unrolls + 1):
            b_idx = cur_idx + td_steps

            if b_idx < len(self.root_values):
                value = self.root_values[b_idx][0] * (self.discount ** td_steps)
            else:
                value = 0

            for i, reward in enumerate(self.reward_history[cur_idx:b_idx]):
                value += reward * (self.discount ** i)

            if 0 < cur_idx <= len(self.reward_history):
                last_reward = self.reward_history[cur_idx - 1]
            else:
                last_reward = 0

            if cur_idx < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[cur_idx]))
                actions.append(self.action_history[cur_idx])
            else:
                num_action = self.action_size
                probs = np.array([1.0 / num_action for _ in range(num_action)])
                targets.append((0, last_reward, probs))
                actions.append(np.random.choice(num_action))
        return targets, actions


class ReplayBuffer(object):
    def __init__(self, config):
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.buffer = deque(maxlen=self.buffer_size)
        self.td_steps = config['td_steps']
        self.unrolls = config['unrolls']

    def save_game(self, game):
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop()
        self.buffer.append(game)

    def sample_batch(self):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_position = [self.sample_position(g) for g in games]

        batch = []
        for (g, i) in zip(games, game_position):
            targets, actions = g.make_target(i, self.unrolls, self.td_steps)
            batch.append(g.state_history[i], actions, targets)

        state_batch, actions_batch, targets_batch = zip(*batch)
        actions_batch = list(zip(*actions_batch))
        targets_init_batch, *targets_recurrent_batch = zip(*targets_batch)
        batch = (state_batch, targets_init_batch, targets_recurrent_batch,
                 actions_batch)

        return batch

    def sample_game(self):
        game = np.random.choice(self.buffer)
        return game

    def sample_position(self, game):
        sampled_idx = np.random.randint(len(game.reward_history) - self.unrolls)
        return sampled_idx


class ActionEmbedding(nn.Module):
    def __init__(self, action_space_size, embedding_size, padding_idx=-1):
        super().__init__()
        self.embedding = nn.Embedding(action_space_size, embedding_size,
                                      padding_idx=padding_idx)

    def forward(self, action):
        return self.embedding(action)


class RepresentationNetwork(nn.Module):
    """
    action embedding이라는 instance 만들어서 Dynamic에도 써야함.
    """
    def __init__(self, action_embedding, data_size, embedding_size,
                 lstm_size, linear_size, latent_size):
        super().__init__()
        # for gate sequence
        self.action_embedding = action_embedding
        self.lstm = nn.LSTM(embedding_size, lstm_size, batch_first=False)

        # for data x_i * x_j
        self.linear = nn.Linear(data_size, linear_size)
        self.relu = nn.ReLU()

        # balance for data & gate input
        self.lstm_norm = nn.LayerNorm(lstm_size)
        self.linear_norm = nn.LayerNorm(linear_size)

        self.combined_fc = nn.Linear(lstm_size+linear_size, latent_size)

    def forward(self, action_sequence_input, data_input):
        embedded = self.action_embedding(action_sequence_input)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.lstm_norm(lstm_out[-1])

        data_linear = self.linear(data_input)
        data_out = self.relu(data_linear)
        data_out = self.linear_norm(data_out)

        combined = torch.cat((lstm_out, data_out), dim=1)
        latent_state = self.combined_fc(combined)

        return latent_state


class ValueNetwork(nn.Module):
    def __init__(self, latent_size, hidden_size, value_support_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, value_support_size)
        )

    def forward(self, x):
        return self.layers(x)


class PolicyNetwork(nn.Module):
    def __init__(self, latent_size, hidden_size, action_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.layers(x)


class DynamicNetwork(nn.Module):
    def __init__(self, action_embedding, latent_size, hidden_size):
        super().__init__()
        self.action_embedding = action_embedding
        input_size = latent_size + action_embedding.embedding.embedding_dim
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.Tanh()
        )

    def forward(self, latent_state, action):
        action_embedding = self.action_embedding(action)
        x = torch.cat((latent_state, action_embedding), dim=-1)
        return self.layers(x)


class RewardNetwork(nn.Module):
    def __init__(self, action_embedding, latent_size, hidden_size):
        super().__init__()
        self.action_embedding = action_embedding
        input_size = latent_size + action_embedding.embedding.embedding_dim
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, latent_state, action):
        action_embedding = self.action_embedding(action)
        x = torch.cat((latent_state, action_embedding), dim=-1)
        return self.layers(x)


class InitialModel(nn.Module):
    def __init__(self, representation_net, value_net, policy_net):
        super().__init__()
        self.representation_net = representation_net
        self.value_net = value_net
        self.policy_net = policy_net

    def forward(self, observation):
        latent_state = self.representation_net(observation)
        value = self.value_net(latent_state)
        policy_logits = self.policy_net(latent_state)
        return latent_state, value, policy_logits


class RecurrentModel(nn.Module):
    def __init__(self, dynamic_net, reward_net, value_net, policy_net):
        super(RecurrentModel, self).__init__()
        self.dynamic_net = dynamic_net
        self.reward_net = reward_net
        self.value_net = value_net
        self.policy_net = policy_net

    def forward(self, latent_state, action):
        latent_state = self.dynamic_net(latent_state, action)
        reward = self.reward_net(latent_state, action)
        value = self.value_net(latent_state)
        policy_logits = self.policy_net(latent_state)
        return latent_state, reward, value, policy_logits


class Networks(nn.Module):
    def __init__(self, representation_net, value_net, policy_net, dynamic_net,
                 reward_net, max_value, action_size):
        super().__init__()
        self.action_size = action_size
        self.representation_network = representation_net
        self.value_network = value_net
        self.policy_network = policy_net
        self.dynamic_network = dynamic_net
        self.reward_network = reward_net

        self.initial_model = InitialModel(self.representation_network,
                                          self.value_network,
                                          self.policy_network)
        self.recurrent_model = RecurrentModel(self.dynamic_network,
                                              self.reward_network,
                                              self.value_network,
                                              self.policy_network)
        self.value_support_size = math.ceil(math.sqrt(max_value)) + 1

    def _value_transform(self, value_support):
        epsilon = 0.001
        value = softmax(value_support)
        value = np.dot(value.detach().numpy(), range(self.value_support_size))
        value = np.sign(value) * (((np.sqrt(1 + 4 * epsilon * (np.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
        return value

    def _reward_transform(self, reward):
        return reward.detach().cpu().numpy()

    def initial_inference(self, observation):
        latent_space, value, policy_logits = self.initial_model(observation)
        assert isinstance(self._value_transform(value), float)
        return self._value_transform(value), 0, policy_logits, latent_space

    def recurrent_inference(self, latent_state, action):
        latent_state, reward, value, policy_logits = self.recurrent_model(latent_state, action)
        return self._value_transform(value), self._reward_transform(reward), \
            policy_logits, latent_state

    def scalar_to_support(self, target_value):
        batch = target_value.size(0)
        targets = torch.zeros((batch, self.value_support_size))
        target_value = torch.sign(target_value) * \
                       (torch.sqrt(torch.abs(target_value) + 1)
                        - 1 + 0.001 * target_value)
        target_value = torch.clamp(target_value, 0, self.value_support_size)
        floor = torch.floor(target_value)
        rest = target_value - floor
        targets[torch.arange(batch, dtype=torch.long), floor.long()] = 1 - rest
        indexes = floor.long() + 1
        mask = indexes < self.value_support_size
        batch_mask = torch.arange(batch)[mask]
        rest_mask = rest[mask]
        index_mask = indexes[mask]
        targets[batch_mask, index_mask] = rest_mask
        return targets


def scale_gradient(tensor, scale):
    return tensor * scale + tensor.detach() * (1. - scale)


def train_network(config, network, replay_buffer, optimizer, train_results):
    for _ in range(config['train_per_epoch']):
        batch = replay_buffer.sample_batch()
        update_weights(config, network, optimizer, batch, train_results)


def update_weights(config, network, optimizer, batch, train_results):
    def loss():
        mse = torch.nn.MSELoss()
        total_reward_loss = 0
        (state_batch, targets_init_batch, targets_recurrent_batch, actions_batch) = batch

        state_batch = torch.tensor(state_batch)

        # get prediction from initial model (i.e. combination of dynamic, value, and policy networks)
        latent_state, initial_values, policy_logits = network.initial_model(state_batch)

        # create a value and policy target from batch data
        target_value_batch, _, target_policy_batch = zip(*targets_init_batch)
        target_value_batch = torch.tensor(target_value_batch).float()
        target_value_batch = network.scalar_to_support(target_value_batch)

        # compute the error for the initial inference
        # reward error is always 0 for initial inference
        value_loss = F.cross_entropy(initial_values, target_value_batch)
        policy_loss = F.cross_entropy(policy_logits,
                                      torch.tensor(target_policy_batch))
        loss = 0.25 * value_loss + policy_loss

        total_value_loss = 0.25 * value_loss.item()
        total_policy_loss = policy_loss.item()

        # unroll batch with recurrent inference and accumulate loss
        for actions_batch, targets_batch in zip(actions_batch, targets_recurrent_batch):
            target_value_batch, target_reward_batch, target_policy_batch = zip(*targets_batch)

            latent_state, rewards, values, policy_logits = network.recurrent_model(latent_state, actions_batch)

            # create a value, policy, and reward target from batch data
            target_value_batch = torch.tensor(target_value_batch).float()
            target_value_batch = network.scalar_to_support(target_value_batch)
            target_policy_batch = torch.tensor(target_policy_batch).float()
            target_reward_batch = torch.tensor(target_reward_batch).float()

            # compute the loss for recurrent_inference
            value_loss = F.cross_entropy(values, target_value_batch)
            policy_loss = F.cross_entropy(policy_logits, target_policy_batch)
            reward_loss = mse(rewards, target_reward_batch)

            # accumulate loss
            loss_step = (0.25 * value_loss + reward_loss + policy_loss)
            total_value_loss += 0.25 * value_loss.item()
            total_policy_loss += policy_loss.item()
            total_reward_loss += reward_loss.item()

            # gradient scaling
            gradient_loss_step = scale_gradient(loss_step, (1 / config['unrolls']))
            loss += gradient_loss_step
            scale = 0.5
            latent_state = latent_state / scale

        # store loss result for plotting
        train_results.total_losses.append(loss.item())
        train_results.value_losses.append(total_value_loss)
        train_results.policy_losses.append(total_policy_loss)
        train_results.reward_losses.append(total_reward_loss)
        return loss

    optimizer.zero_grad()
    loss = loss()
    loss.backward()
    optimizer.step()


class MCTS:
    def __init__(self, config):
        self.config = config

    def run_mcts(self, config, root, network, min_max_stats):
        for i in range(config['num_simulations']):
            history = []
            node = root
            search_path = [node]

            # expand node until reaching the leaf node
            while node.expanded:
                action, node = self.select_child(config, node, min_max_stats)
                history.append(action)
                search_path.append(node)
            parent = search_path[-2]
            action = history[-1]

            # expand the leaf node
            value = self.expand_node(node, list(
                range(config['action_space_size'])), network, parent.latent_state, action)

            # perform backpropagation
            self.backpropagate(search_path, value, config['discount'], min_max_stats)

    def select_action(self, config, node, test=False):
        visit_counts = [(child.visit_count, action) for action, child in
            node.children.items()]
        if not test:
            t = config['visit_softmax_temperature_fn']
            action = self.softmax_sample(visit_counts, t)
        else:
            action = self.softmax_sample(visit_counts, 0)
        return action

    def select_child(self, config, node, min_max_stats):
        best_action, best_child = None, None
        ucb_compare = -np.inf
        for action, child in node.children.items():
            ucb = self.ucb_score(config, node, child, min_max_stats)
            if ucb > ucb_compare:
                ucb_compare = ucb
                best_action = action  # action, int
                best_child = child  # node object
        return best_action, best_child

    def ucb_score(self, config, parent, child, min_max_stats):
        pb_c = np.log((parent.visit_count + config['pb_c_base'] + 1)
                      / config['pb_c_base']) + config['pb_c_init']
        pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior.detach().numpy()
        if child.visit_count > 0:
            value_score = min_max_stats.normalize(
                child.reward + config['discount'] * child.value())
        else:
            value_score = 0
        return prior_score + value_score

    def expand_root(self, node, actions, network, observation):
        # obtain the latent state, policy, and value of the root node
        # by using a InitialModel
        observation = torch.tensor(observation)
        transformed_value, reward, policy_logits, latent_state = network.initial_inference(
            observation)
        node.latent_state = latent_state
        node.reward = reward  # always 0 for initial inference

        # extract softmax policy and set node.policy
        softmax_policy = softmax(torch.squeeze(policy_logits))
        node.policy = softmax_policy

        # instantiate node's children with prior values, obtained from the predicted policy
        for action, prob in zip(actions, softmax_policy):
            child = Node(prob)
            node.children[action] = child

        # set node as expanded
        node.expanded = True

        return transformed_value

    def expand_node(self, node, actions, network, parent_state, parent_action):
        # run recurrent inference at the leaf node
        transformed_value, reward, policy_logits, latent_state = network.recurrent_inference(
            parent_state, parent_action)
        node.latent_state = latent_state
        node.reward = reward

        # compute softmax policy and store it to node.policy
        softmax_policy = softmax(torch.squeeze(policy_logits))
        node.policy = softmax_policy

        # instantiate node's children with prior values, obtained from the predicted softmax policy
        for action, prob in zip(actions, softmax_policy):
            child = Node(prob)
            node.children[action] = child

        # set node as expanded
        node.expanded = True

        return transformed_value

    def add_exploration_noise(self, config, node):
        actions = list(node.children.keys())
        noise = np.random.dirichlet([config['root_dirichlet_alpha']] * len(actions))
        frac = config['root_exploration_fraction']
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    def backpropagate(self, path, value, discount, min_max_stats):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            min_max_stats.update(node.value())
            value = node.reward + discount * value

    def softmax_sample(self, visit_counts, temperature):
        counts_arr = np.array([c[0] for c in visit_counts])
        if temperature == 0:  # argmax
            action_idx = np.argmax(counts_arr)
        else:  # softmax
            numerator = np.power(counts_arr, 1 / temperature)
            denominator = np.sum(numerator)
            dist = numerator / denominator
            action_idx = np.random.choice(np.arange(len(counts_arr)), p=dist)

        return action_idx



class MinMaxStats(object):
    def __init__(self, minimum, maximum):
        self.maximum = maximum
        self.minimum = minimum

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value



class TrainResults(object):
    def __init__(self):
        self.value_losses = []
        self.reward_losses = []
        self.policy_losses = []
        self.total_losses = []

    def plot_total_loss(self):
        x = np.arange(len(self.total_losses))
        plt.figure()
        plt.plot(x, self.total_losses, label="Train Loss", color='k')
        plt.xlabel("Train Steps", fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        plt.show()
        # plt.savefig('./RL/ModelBasedML/figure/total_loss.png')

    def plot_individual_losses(self):
        x = np.arange(len(self.total_losses))
        plt.figure()
        plt.plot(x, self.value_losses, label="Value Loss", color='r')
        plt.plot(x, self.policy_losses, label="Policy Loss", color='b')
        plt.plot(x, self.reward_losses, label="Reward Loss", color='g')
        plt.xlabel("Train Episode", fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        plt.legend()
        plt.show()
        # plt.savefig('./RL/ModelBasedML/figure/individual_loss.png')



class TestResults(object):

    def __init__(self):
        self.test_rewards = []

    def add_reward(self, reward):
        self.test_rewards.append(reward)

    def plot_rewards(self):
        x = np.arange(len(self.test_rewards))
        plt.subplots()
        plt.plot(x, self.test_rewards, label="Test Reward", color='orange')
        plt.xlabel("Test Episode", fontsize=15)
        plt.ylabel("Reward", fontsize=15)
        plt.show()
        # plt.savefig('./RL/ModelBasedML/figure/test_reward.png')



def self_play(env, config, replay_buffer, network):
    # create objects to store data for plotting
    test_rewards = TestResults()
    train_results = TrainResults()

    # create optimizer for training
    optimizer = torch.optim.Adam(network.parameters(), lr=config['lr_init'])

    # self-play and network training iterations
    for i in range(config['num_epochs']):  # Number of Steps of train/play alternations
        print(f"===Epoch Number {i}===")
        score = play_games(config, replay_buffer, network, env)
        print("Average traininig score:", score)
        train_network(config, network, replay_buffer, optimizer, train_results)
        print("Average test score:", test(config, network, env, test_rewards))

    # plot
    train_results.plot_individual_losses()
    train_results.plot_total_loss()
    test_rewards.plot_rewards()



def play_games(config, replay_buffer, network, env):
    """
    Play multiple games and store them in the replay buffer
    """
    returns = 0

    for _ in range(config['games_per_epoch']):
        game = play_game(config, network, env)
        replay_buffer.save_game(game)
        returns += sum(game.reward_history)

    return returns / config['games_per_epoch']




def play_game(config, network: Networks, env):
    """
    Plays one game
    """
    # Initialize environment
    start_state = env.reset()

    game = Game(config['action_space_size'], config['discount'], start_state)
    mcts = MCTS(config)

    # Play a game using MCTS until game will be done or max_depth will be reached
    while not game.done and len(game.action_history) < config['max_depth']:
        root = Node(0)

        # Create MinMaxStats Object to normalize values
        min_max_stats = MinMaxStats(config['min_value'], config['max_value'])

        # Expand the current root node
        curr_state = game.current_state
        value = mcts.expand_root(root, list(range(config['action_space_size'])),
                                 network, curr_state)
        mcts.backpropagate([root], value, config['discount'], min_max_stats)
        mcts.add_exploration_noise(config, root)

        # Run MCTS
        mcts.run_mcts(config, root, network, min_max_stats)

        # Select an action to take
        action = mcts.select_action(config, root)

        # Take an action and store tree search statistics
        game.take_action(action, env)
        game.store_search_stats(root)
    print(f'Total reward for a train game: {sum(game.reward_history)}')
    return game



def test(config, network, env, test_rewards):
    """
    Test performance using trained networks
    """
    mcts = MCTS(config)
    returns = 0
    for _ in range(config['episodes_per_test']):
        # env.seed(1) # use for reproducibility of trajectories
        start_state, _ = env.reset()
        game = Game(config['action_space_size'], config['discount'],
                    start_state)
        while not game.done and len(game.action_history) < config['max_depth']:
            min_max_stats = MinMaxStats(config['min_value'],
                                        config['max_value'])
            curr_state = game.current_state
            root = Node(0)
            value = mcts.expand_root(root,
                                     list(range(config['action_space_size'])),
                                     network, curr_state)
            # don't run mcts.add_exploration_noise for test
            mcts.backpropagate([root], value, config['discount'], min_max_stats)
            mcts.run_mcts(config, root, network, min_max_stats)
            action = mcts.select_action(config, root,
                                        test=True)  # argmax action selection
            game.take_action(action, env)
        total_reward = sum(game.reward_history)
        print(f'Total reward for a test game: {total_reward}')
        test_rewards.add_reward(total_reward)
        returns += total_reward
    return returns / config['episodes_per_test']


SEED = 42

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


action_mapping = {
    0: 'X',
    1: 'Y',
    2: 'Z',
    3: 'CX',
    4: 'CY',
    5: 'CZ',
    # CRx(π/n)
    6: ('CRx_pi_over_n', 1),
    7: ('CRx_pi_over_n', 2),
    8: ('CRx_pi_over_n', 3),
    9: ('CRx_pi_over_n', 4),
    10: ('CRx_pi_over_n', 8),
    # CRy(π/n)
    11: ('CRy_pi_over_n', 1),
    12: ('CRy_pi_over_n', 2),
    13: ('CRy_pi_over_n', 3),
    14: ('CRy_pi_over_n', 4),
    15: ('CRy_pi_over_n', 8),
    # CRz(π/n)
    16: ('CRz_pi_over_n', 1),
    17: ('CRz_pi_over_n', 2),
    18: ('CRz_pi_over_n', 3),
    19: ('CRz_pi_over_n', 4),
    20: ('CRz_pi_over_n', 8),
    # Rx(πx), Ry(πx), Rz(πx)
    21: 'Rx_pi_x',
    22: 'Ry_pi_x',
    23: 'Rz_pi_x',
    # Rx(π/n)
    24: ('Rx_pi_over_n', 1),
    25: ('Rx_pi_over_n', 2),
    26: ('Rx_pi_over_n', 3),
    27: ('Rx_pi_over_n', 4),
    28: ('Rx_pi_over_n', 8),
    # Ry(π/n)
    29: ('Ry_pi_over_n', 1),
    30: ('Ry_pi_over_n', 2),
    31: ('Ry_pi_over_n', 3),
    32: ('Ry_pi_over_n', 4),
    33: ('Ry_pi_over_n', 8),
    # Rz(π/n)
    34: ('Rz_pi_over_n', 1),
    35: ('Rz_pi_over_n', 2),
    36: ('Rz_pi_over_n', 3),
    37: ('Rz_pi_over_n', 4),
    38: ('Rz_pi_over_n', 8),
    # Rx(arctan(x)), Ry(arctan(x)), Rz(arctan(x))
    39: 'Rx_arctan_x',
    40: 'Ry_arctan_x',
    41: 'Rz_arctan_x',
    42: 'H',
}


class QASEnv:
    def __init__(self, max_depth):
        super().__init__()
        self.max_depth = max_depth
        self.reset()

    def reset(self):
        self.actions = []  # List of actions taken (gate indices)
        self.depth = 0  # Current circuit depth
        return self.get_observation()

    def get_observation(self):
        obs = torch.zeros(self.max_depth, dtype=torch.long)
        obs[:len(self.actions)] = torch.tensor(self.actions, dtype=torch.long)
        return obs

    def step(self, action):
        self.actions.append(action)
        self.depth += 1
        done = self.depth >= self.max_depth
        observation = self.get_observation()

        return done, observation

    def get_circuit(self):
        return self.actions


def get_fidelity(actions, x_1, x_2, y):
    @qml.qnode(dev)
    def circuit(inputs):
        build_circuit(inputs[0:4], actions)
        qml.adjoint(build_circuit)(inputs[4:8], actions)
        return qml.probs(wires=range(n_qubit))

    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes={})
    x = torch.concat([x_1, x_2], 1)
    pred = qlayer(x)[:, 0]
    fidelity_loss = torch.nn.MSELoss()(pred, y)
    reward = 1 - fidelity_loss.item()
    return reward


def build_circuit(x_data, actions):
    for action in actions:
        gate_info = action_mapping[action]
        apply_gate(gate_info, x_data)


def apply_gate(gate_info, x_data):
    wires = range(n_qubit)
    if isinstance(gate_info, str):
        gate = gate_info
        n = None
    else:
        gate, n = gate_info
    if gate == 'X':
        for qubit in wires:
            qml.PauliX(wires=qubit)
    elif gate == 'Y':
        for qubit in wires:
            qml.PauliY(wires=qubit)
    elif gate == 'Z':
        for qubit in wires:
            qml.PauliZ(wires=qubit)
    elif gate == 'CX':
        for qubit in range(n_qubit - 1):
            qml.CNOT(wires=[qubit, qubit + 1])
    elif gate == 'CY':
        for qubit in range(n_qubit - 1):
            qml.CY(wires=[qubit, qubit + 1])
    elif gate == 'CZ':
        for qubit in range(n_qubit - 1):
            qml.CZ(wires=[qubit, qubit + 1])
    elif gate == 'CRx_pi_over_n':
        angle = np.pi / n
        for qubit in range(n_qubit - 1):
            qml.CRX(angle, wires=[qubit, qubit + 1])
    elif gate == 'CRy_pi_over_n':
        angle = np.pi / n
        for qubit in range(n_qubit - 1):
            qml.CRY(angle, wires=[qubit, qubit + 1])
    elif gate == 'CRz_pi_over_n':
        angle = np.pi / n
        for qubit in range(n_qubit - 1):
            qml.CRZ(angle, wires=[qubit, qubit + 1])
    elif gate == 'Rx_pi_x':
        for qubit in wires:
            qml.RX(np.pi * x_data[qubit], wires=qubit)
    elif gate == 'Ry_pi_x':
        for qubit in wires:
            qml.RY(np.pi * x_data[qubit], wires=qubit)
    elif gate == 'Rz_pi_x':
        for qubit in wires:
            qml.RZ(np.pi * x_data[qubit], wires=qubit)
    elif gate == 'Rx_pi_over_n':
        angle = np.pi / n
        for qubit in wires:
            qml.RX(angle, wires=qubit)
    elif gate == 'Ry_pi_over_n':
        angle = np.pi / n
        for qubit in wires:
            qml.RY(angle, wires=qubit)
    elif gate == 'Rz_pi_over_n':
        angle = np.pi / n
        for qubit in wires:
            qml.RZ(angle, wires=qubit)
    elif gate == 'Rx_arctan_x':
        for qubit in wires:
            qml.RX(np.arctan(x_data[qubit]), wires=qubit)
    elif gate == 'Ry_arctan_x':
        for qubit in wires:
            qml.RY(np.arctan(x_data[qubit]), wires=qubit)
    elif gate == 'Rz_arctan_x':
        for qubit in wires:
            qml.RZ(np.arctan(x_data[qubit]), wires=qubit)
    elif gate == 'H':
        for qubit in wires:
            qml.Hadamard(wires=qubit)



if __name__ == "__main__":

    action_space_size = len(action_mapping)

    config = {
        # Quantum Specific
        'n_qubit': 4,
        'data_size': 4 * 2,

        # Environment
        'action_space_size': action_space_size,  # number of action

        # Simulation
        'games_per_epoch': 20,
        'num_epochs': 25,
        'train_per_epoch': 30,
        'episodes_per_test': 10,

        'visit_softmax_temperature_fn': 1,
        'max_depth': 10,  # circuit depth
        'num_simulations': 20,
        'discount': 0.99,
        'min_value': 0,
        'max_value': 200,

        # Root prior exploration noise.
        'root_dirichlet_alpha': 0.2,
        'root_exploration_fraction': 0.25,

        # UCB parameters
        'pb_c_base': 19652,
        'pb_c_init': 1.25,

        # Model fitting config
        'embedding_size': action_space_size,
        'lstm_size': 32,
        'linear_size': 32,
        'hidden_size': 32,
        'latent_size': action_space_size,
        'buffer_size': 100,
        'batch_size': 64,
        'unrolls': 3,
        'td_steps': 3,
        'lr_init': 0.005,
    }

    dev = qml.device('default.qubit', wires=n_qubit)




    value_support_size = math.ceil(math.sqrt(config['max_value'])) + 1


    # Set seeds for reproducibility
    set_seeds()

    # Create networks
    action_embed = ActionEmbedding(action_space_size=action_space_size, embedding_size=config['embedding_size'], padding_idx=-1)
    rep_net = RepresentationNetwork(action_embedding=action_embed, data_size=config['data_size'], embedding_size=config['embedding_size'], lstm_size=config['lstm_size'], linear_size=config['linear_size'], latent_size=config['latent_size'])
    val_net = ValueNetwork(latent_size=config['latent_size'], hidden_size=config['hidden_size'], value_support_size=value_support_size)
    pol_net = PolicyNetwork(latent_size=config['latent_size'], hidden_size=config['hidden_size'], action_size=action_space_size)
    dyn_net = DynamicNetwork(action_embedding=action_embed, latent_size=config['latent_size'], hidden_size=config['hidden_size'])
    rew_net = RewardNetwork(action_embedding=action_embed, latent_size=config['latent_size'], hidden_size=config['hidden_size'])
    network = Networks(representation_net=rep_net, value_net=val_net, policy_net=pol_net, dynamic_net=dyn_net, reward_net=rew_net, max_value=config['max_value'], action_size=action_space_size)



    # Create environment
    env = QASEnv(max_depth=config['max_depth'])

    # Create buffer to store games
    replay_buffer = ReplayBuffer(config)

    self_play(env, config, replay_buffer, network)
