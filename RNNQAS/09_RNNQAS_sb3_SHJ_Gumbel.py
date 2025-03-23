import glob
import os
from datetime import datetime
from itertools import permutations

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn

from data import data_load_and_process as dataprep
from data import new_data
from utils import make_arch_sb3_SHJ

dev = qml.device("default.qubit", wires=4)


class GumbelSoftmaxPolicy(ActorCriticPolicy):
    """
    기존 ActorCriticPolicy를 상속받아, 액션 샘플링 시 Gumbel-Softmax를 사용하여
    미분가능한 근사(discrete decision의 continuous relaxation)를 도입합니다.
    """
    def __init__(self, *args, gumbel_tau: float = 1.0, **kwargs):
        # gumbel_tau: 온도 파라미터 (낮을수록 hard sample에 가까워짐)
        super(GumbelSoftmaxPolicy, self).__init__(*args, **kwargs)
        self.gumbel_tau = gumbel_tau

    def forward(self, obs, deterministic: bool = False):
        """
        obs: 환경으로부터 받은 관측값
        deterministic: True인 경우 확률 분포의 argmax를 사용하여 결정적 행동 선택
        반환: (actions, values, log_prob)
        """
        # 1. 특징 추출
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)  # (batch_size, num_actions)

        # 2. 액션 선택 및 log_prob 계산
        if deterministic:
            # 확률 분포의 argmax 선택 (결정적 선택)
            action_probs = F.softmax(logits, dim=1)
            actions = action_probs.argmax(dim=1)
            # 선택된 액션의 log probability 계산
            log_prob = F.log_softmax(logits, dim=1).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        else:
            # Gumbel-Softmax를 사용한 미분가능한 샘플링 (hard sample 반환)
            action_onehot = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True)
            actions = action_onehot.argmax(dim=1)
            # 원래 logits에서 log_softmax 계산 후, 선택된 액션에 해당하는 값 추출
            log_prob = F.log_softmax(logits, dim=1).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # 3. 가치 함수 계산
        values = self.value_net(latent_vf)

        return actions, values, log_prob

    def _predict(self, obs, deterministic: bool = False):
        # _predict에서는 일반적으로 액션만 반환하면 됩니다.
        actions, _, _ = self.forward(obs, deterministic=deterministic)
        return actions


def parse_eventfile(event_file, scalar_key="train/policy_gradient_loss"):
    """
    특정 이벤트 파일(.tfevents)에서 scalar_key에 해당하는 스칼라 데이터를 (step[], value[])로 파싱
    """
    ea = EventAccumulator(event_file)
    ea.Reload()  # 실제 파일 로드

    # event_accumulator.Tags() => {"scalars": [...], "images": [...], ...} 식으로 태그 목록
    if scalar_key not in ea.Tags()["scalars"]:
        print(f"Warning: '{scalar_key}' not found in {event_file}")
        return [], []

    scalar_list = ea.Scalars(scalar_key)
    steps = [scalar.step for scalar in scalar_list]
    vals = [scalar.value for scalar in scalar_list]
    return steps, vals


def plot_policy_loss(log_dir, output_filename="policy_loss_plot.png"):
    """
    log_dir 아래 있는 이벤트 파일(.tfevents)을 찾아서,
    'train/policy_gradient_loss' 스칼라를 파싱 후, 라인 플롯으로 저장
    """
    # 이벤트 파일을 전부 찾기
    event_files = glob.glob(os.path.join(log_dir, "**/events.out.tfevents.*"), recursive=True)
    if len(event_files) == 0:
        print("No event files found in", log_dir)
        return

    # 여기서는 편의상 '가장 마지막'에 생성된 이벤트 파일을 사용
    # (원하는 파일을 지정하거나, 여러 파일을 합쳐 그려도 됨)
    event_files.sort(key=os.path.getmtime)
    target_event_file = event_files[-1]

    steps, vals = parse_eventfile(target_event_file, scalar_key="train/policy_gradient_loss")
    if len(steps) == 0:
        print("No data found for 'train/policy_gradient_loss'")
        return

    plt.figure()
    plt.plot(steps, vals, label="policy_gradient_loss")
    plt.xlabel("Timesteps")
    plt.ylabel("Loss")
    plt.title("Policy Gradient Loss over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved policy loss plot => {output_filename}")


def plot_nqe_loss(loss_values, filename="fidelity_loss.png"):
    """
    Plot the NQE validation loss and save as a PNG file.

    Parameters:
        loss_values (list of float): List of loss values.
        filename (str): Name of the file to save the plot.
    """
    episodes = range(1, len(loss_values) + 1)

    # 그래프 생성
    plt.figure()
    plt.plot(episodes, loss_values, marker='o', linestyle='-', label='Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Last Step Loss')
    plt.grid(True)
    plt.legend()

    # 그래프 저장
    plt.savefig(filename)



def get_fidelity(structure, batch_size, X_batch, y_batch):
    def quantum_embedding(x):
        for gate_type, control, target, data_dim in structure:
            if gate_type == 0:
                qml.RX(x[data_dim], wires=control)
            elif gate_type == 1:
                qml.RY(x[data_dim], wires=control)
            elif gate_type == 2:
                qml.RZ(x[data_dim], wires=control)
            elif gate_type == 3:
                qml.CNOT(wires=[control, target])

    @qml.qnode(dev, interface='torch')
    def circuit(inputs):
        quantum_embedding(inputs[0:4])
        qml.adjoint(quantum_embedding)(inputs[4:8])

        return qml.probs(wires=range(4))

    loss_fn = torch.nn.MSELoss()
    X1_batch, X2_batch, Y_batch = new_data(batch_size, X_batch, y_batch)
    qlayer1 = qml.qnn.TorchLayer(circuit, weight_shapes={})
    x = torch.concat([X1_batch, X2_batch], 1)
    x = qlayer1(x)
    pred = x[:, 0]
    loss = loss_fn(pred, Y_batch)
    return loss.item()


class QASEnv(gym.Env):
    def __init__(self, num_qubit, num_gate_class, max_gate, batch_size, done_criteria,
                 X_train, Y_train, X_test, Y_test):
        super().__init__()
        self.num_qubit = num_qubit
        self.num_gate_class = num_gate_class  # RX, RT, RZ, CNOT(targ), CNOT(ctrl)
        self.max_gate = max_gate

        self.batch_size = batch_size
        self.done_criteria = done_criteria

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.action_space = gym.spaces.Discrete((num_gate_class - 1) * (num_qubit * (num_qubit - 1)) * num_qubit)
        self.observation_space = gym.spaces.Box(
            low=0, high=5, shape=(max_gate, num_qubit, num_gate_class), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gate_step = 0
        self.structure_list = []

        state = torch.zeros((self.max_gate, self.num_qubit, self.num_gate_class), dtype=torch.float32)

        return state.numpy(), {}

    def action_to_structure(self, action):
        gate_type = action // (self.num_qubit * (self.num_qubit - 1) * self.num_qubit)
        remainder = action % (self.num_qubit * (self.num_qubit - 1) * self.num_qubit)

        qubit_idx = remainder // self.num_qubit
        control, target = list(permutations(range(self.num_qubit), 2))[qubit_idx]

        data_dim = remainder % self.num_qubit

        return gate_type, control, target, data_dim

    def step(self, action):
        gate_type, control, target, data_dim = self.action_to_structure(action)
        self.structure_list.append((gate_type, control, target, data_dim))

        state = make_arch_sb3_SHJ(self.structure_list, self.max_gate, self.num_qubit, self.num_gate_class)

        loss = get_fidelity(self.structure_list, self.batch_size, self.X_train, self.Y_train)
        reward = 1 - loss

        self.gate_step += 1
        print(f"step: {self.gate_step} and loss: {loss}")
        done = loss < self.done_criteria or self.gate_step >= self.max_gate

        info = {"valid_loss": loss}

        return state, reward, done, {}, info


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


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_valid_losses = []
        self.episode_actions = []

        self.current_actions = []
        self.current_valid_loss = []
        self.current_reward_sum = 0

        self.done_count = 0
        self.early_finish_criteria = 10

    def _on_step(self) -> bool:
        actions = self.locals["actions"]
        infos = self.locals["infos"]

        action = actions[0]
        info = infos[0]

        self.current_actions.append(action)

        if "valid_loss" in info:
            self.current_valid_loss.append(info["valid_loss"])

        if "episode" in info:
            # 이 에피소드의 보상 합
            ep_reward = info["episode"]["r"]
            self.episode_rewards.append(ep_reward)

            # 이 에피소드의 valid_loss는 스텝별로 여러 개가 있을 수 있는데,
            # 여기서는 마지막 스텝의 값을 대표값으로 쓰거나, 평균을 쓰거나 자유롭게 정의 가능
            if len(self.current_valid_loss) > 0:
                ep_valid_loss = self.current_valid_loss[-1]
            else:
                ep_valid_loss = None
            self.episode_valid_losses.append(ep_valid_loss)

            # actions 기록 (리스트 통째로)
            self.episode_actions.append(self.current_actions)

            # Done 체크
            if info.get("done", False):
                self.done_count += 1
            else:
                self.done_count = 0

            # Early stopping 조건 확인
            if self.done_count >= self.early_finish_criteria:
                if self.verbose > 0:
                    print("[Callback] Early finish triggered: done occurred 10 times consecutively.")
                return False  # 학습 종료

            # 로그를 찍어볼 수도 있음
            if self.verbose > 0:
                print(
                    f"[Callback] End of episode #{len(self.episode_rewards)}: Reward={ep_reward:.3f}, ValidLoss={ep_valid_loss}")

            # 에피소드가 끝났으므로, 임시 버퍼 초기화
            self.current_actions = []
            self.current_valid_loss = []
            self.current_reward_sum = 0

        return True

    def _on_training_end(self):
        if self.verbose > 0:
            print("========== Training finished! ==========")
            print(f"Total episodes: {len(self.episode_rewards)}")

if __name__ == "__main__":
    print(datetime.now())
    num_qubit = 4
    num_gate_class = 5
    max_gate = 20

    max_epoch = 50  # 100
    batch_size = 200

    X_train, X_test, Y_train, Y_test = dataprep(dataset='kmnist', reduction_sz=num_qubit)
    done_criteria = 0.1

    env = QASEnv(
        num_qubit=num_qubit,
        num_gate_class=num_gate_class,
        max_gate=max_gate,
        batch_size=batch_size,
        done_criteria=done_criteria,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
    )

    monitored_env = Monitor(env)

    policy_kwargs = dict(
        features_extractor_class=CNN_LSTM_Extractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO(
        policy=GumbelSoftmaxPolicy,
        env=monitored_env,
        n_steps=128,
        gamma=0.95,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)
    custom_callback = CustomCallback(verbose=2)

    print('Learning Start...')
    model.learn(total_timesteps=max_epoch * max_gate, callback=custom_callback)
    model.save('test')
    plot_nqe_loss(custom_callback.episode_valid_losses, filename="fidelity_loss.png")
    plot_policy_loss(log_dir="./logs", output_filename="policy_loss_plot.png")
    print(datetime.now())
