import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from data import data_load_and_process as dataprep
from data import new_data
from model import NQEModel
from utils import make_arch_sb3, generate_layers, set_done_loss


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


def plot_nqe_loss(loss_values, filename="NQE_loss.png"):
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
    plt.title('Last Step NQE Validation Loss')
    plt.grid(True)
    plt.legend()

    # 그래프 저장
    plt.savefig(filename)


def preprocess_done_trajectory(trajectories):
    """
    길이가 다른 리스트를 가장 긴 리스트의 길이에 맞춰 마지막 값을 복사하여 확장합니다.
    """
    max_length = max(len(sublist) for sublist in trajectories)  # 가장 긴 리스트의 길이
    processed_li = []

    for sublist in trajectories:
        if len(sublist) < max_length:  # 길이가 짧으면
            last_value = 0  # 마지막 값 복사 (빈 리스트인 경우 0 추가)
            sublist = sublist + [last_value] * (max_length - len(sublist))  # 길이 맞추기
        processed_li.append(sublist)

    return processed_li


def save_trajectory(nested_list, filename="trajectory_plot.png", max_epoch_PG=200, num_layer=64):
    """
    Plots and saves a trajectory graph based on the provided data.
    Parameters:
        li (dict): Trajectory data where keys are timesteps and values contain 'layer_list'.
        filename (str): The name of the file to save the plot (default is "trajectory_plot.png").
        total_timesteps (int): Total number of timesteps to display on the x-axis.
        total_positions (int): Total number of positions (y-axis range).
    """
    colors = ['red', 'yellow', 'green', 'skyblue', 'black',
              'blue', 'orange', 'purple', 'pink', 'brown',
              'gray', 'tan', 'blue']
    # x축: timestep, y축: value
    timesteps = list(range(len(nested_list)))
    trajectories = preprocess_done_trajectory(nested_list)

    # 확대된 그래프 생성
    ratio = max(int(max_epoch_PG/num_layer), 1)
    plt.figure(figsize=(20 * ratio, 20))  # 커다란 그림

    for i in range(len(trajectories[0])):  # 위치 수만큼 반복
        y_values = [trajectory[i] for trajectory in trajectories]  # i번째 위치의 값 추출
        plt.plot(timesteps, y_values, marker='o', label=f'Trajectory {i + 1}', color=colors[i % len(colors)])
    # x축과 y축을 grid로 표시
    plt.xticks(range(0, max_epoch_PG + 1), fontsize=10)  # 10 간격의 xticks
    plt.yticks(range(num_layer), fontsize=10)  # 모든 y축 index 표시
    plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    # 그래프 속성 설정
    plt.title("Trajectory Plot", fontsize=16)
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Position (Layer Index)", fontsize=14)
    plt.legend(title="Trajectories", fontsize=12, title_fontsize=14)
    # 그래프 저장
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


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

        state = torch.zeros((self.max_layer_step * 4, self.num_qubit, self.num_gate_class), dtype=torch.float32)

        return state.numpy(), {}

    def step(self, action):
        self.layer_list.append(action)
        gate_list = [item for i in self.layer_list for item in self.layer_set[int(i)]]
        state = make_arch_sb3(gate_list, self.num_qubit, self.max_layer_step, self.num_gate_class)

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
        print(f"layer_step: {self.layer_step} and loss: {loss}")
        done = loss < self.done_criteria or self.layer_step >= self.max_layer_step

        info = {"valid_loss": loss.item()}

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
    num_layer = 64
    max_layer_step = 10

    lr_NQE = 0.01
    max_epoch_PG = 300  # 50
    max_epoch_NQE = 50  # 50
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

    monitored_env = Monitor(env)

    policy_kwargs = dict(
        features_extractor_class=CNN_LSTM_Extractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO(
        policy="MlpPolicy",
        env=monitored_env,
        n_steps=128,
        gamma=0.9,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)
    custom_callback = CustomCallback(verbose=2)

    print('Learning Start...')
    model.learn(total_timesteps=max_epoch_PG * max_layer_step, callback=custom_callback)
    model.save('test')
    plot_policy_loss(log_dir="./logs", output_filename="policy_loss_plot.png")
    plot_nqe_loss(custom_callback.episode_valid_losses, filename="NQE_loss.png")
    save_trajectory(custom_callback.episode_actions, filename="trajectory_plot.png",
                    max_epoch_PG=max_epoch_PG, num_layer=num_layer)
    print(datetime.now())
