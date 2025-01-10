from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from data import data_load_and_process as dataprep
from data import new_data
from model import NQEModel
from utils import make_arch_batchless, generate_layers, set_done_loss


class QASEnv(gym.Env):
    def __init__(self, num_qubit, num_gate_class, num_layer, max_layer_step,
                 lr_NQE, max_epoch_NQE, batch_size, layer_set, baseline, done_criteria,
                 X_train, Y_train, X_test, Y_test):
        super().__init__()
        self.num_qubit = num_qubit
        self.num_gate_class = num_gate_class
        self.num_layer = num_layer
        self.max_layer_step = max_layer_step

        self.lr_NQE = lr_NQE
        self.max_epoch_NQE = max_epoch_NQE
        self.batch_size = batch_size
        self.layer_set = layer_set
        self.baseline = baseline
        self.done_criteria = done_criteria

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.action_space = gym.spaces.Discrete(num_layer)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(max_layer_step * 4, num_qubit, num_gate_class), dtype=np.float32
        )

        self.loss_fn = nn.MSELoss()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.layer_step = 0
        self.layer_list = []

        state = torch.randint(0, 1, (1, self.num_qubit, self.num_gate_class)).float()

        return state.numpy(), {}

    def step(self, action):
        self.layer_list.append(action)
        gate_list = [item for i in self.layer_list for item in self.layer_set[int(i)]]
        state = make_arch_batchless(gate_list, self.num_qubit)

        NQE_model = NQEModel(gate_list)
        NQE_model.train()
        NQE_opt = torch.optim.SGD(NQE_model.parameters(), lr=self.lr_NQE)

        for _ in range(self.max_epoch_NQE):
            X1_batch, X2_batch, Y_batch = new_data(self.batch_size, self.X_train, self.Y_train)
            pred = NQE_model(X1_batch, X2_batch)
            loss = self.loss_fn(pred, Y_batch)

            NQE_opt.zero_grad()
            loss.backward()
            NQE_opt.step()

        valid_loss_list = []
        NQE_model.eval()
        for _ in range(self.batch_size):
            X1_batch, X2_batch, Y_batch = new_data(self.batch_size, self.X_test, self.Y_test)
            with torch.no_grad():
                pred = NQE_model(X1_batch, X2_batch)
            valid_loss_list.append(self.loss_fn(pred, Y_batch))

        loss = sum(valid_loss_list) / self.batch_size
        reward = 1 - loss - self.baseline

        self.layer_step += 1
        done = loss < self.done_criteria or self.layer_step >= self.max_layer_step

        return state, reward, done, {}, {}


class CNN_LSTM_Extractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 features_dim: int = 128,
                 hidden_dim: int = 32,
                 num_layers: int = 1):
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, features_dim)

    def forward(self, observations: torch.Tensor):
        batch_size, seq_len, h, w = observations.shape
        observations = observations.view(batch_size * seq_len, 1, h, w)

        cnn_feat = self.cnn(observations)
        cnn_feat = cnn_feat.mean(dim=[2, 3])
        cnn_feat = cnn_feat.view(batch_size, seq_len, -1)

        out, (h_n, c_n) = self.lstm(cnn_feat)
        last_hidden = h_n[-1]
        features = self.fc(last_hidden)

        return features


if __name__ == "__main__":
    print(datetime.now())
    num_qubit = 4
    num_gate_class = 5
    num_layer = 64
    max_layer_step = 7

    lr_NQE = 0.01
    max_epoch_PG = 7  # 50
    max_epoch_NQE = 6  # 50
    batch_size = 25

    lr_PG = 0.001

    layer_set = generate_layers(num_qubit, num_layer)
    X_train, X_test, Y_train, Y_test = dataprep(dataset='kmnist', reduction_sz=num_qubit)
    baseline, done_criteria = set_done_loss(max_layer_step, num_qubit, max_epoch_NQE, batch_size,
                                            X_train, Y_train, X_test, Y_test)

    env = QASEnv(
        num_qubit=num_qubit,
        num_gate_class=num_gate_class,
        num_layer=num_layer,
        max_layer_step=max_layer_step,
        lr_NQE=lr_NQE,
        max_epoch_NQE=max_epoch_NQE,
        batch_size=batch_size,
        layer_set=layer_set,
        baseline=baseline,
        done_criteria=done_criteria,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
    )

    policy_kwargs = dict(
        features_extractor_class=CNN_LSTM_Extractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    print('Learning Start...')
    model.learn(total_timesteps=max_epoch_PG * max_layer_step)
    print(datetime.now())
