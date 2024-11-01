import random
from collections import deque, namedtuple

import torch
from torch import nn

from agent import QASEnv
from data import new_data

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, num_of_qubits):
        super(DQNNetwork, self).__init__()
        self.state_linear_relu_stack = nn.Sequential(
            nn.Linear(state_size * 4, state_size * 8),
            nn.ReLU(),
            nn.Linear(state_size * 8, state_size * 4),
        )
        # 각 큐빗에 대한 Q-값을 출력하는 레이어
        self.action_value_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(state_size * 4, action_size * 2),
                nn.ReLU(),
                nn.Linear(action_size * 2, action_size),
            ) for _ in range(num_of_qubits)]
        )

    def forward(self, state):
        # state: [batch_size, state_size * 4]
        state_new = self.state_linear_relu_stack(state)
        q_values = []
        for qubit_action_value_layer in self.action_value_layers:
            q_value = qubit_action_value_layer(
                state_new)  # [batch_size, action_size]
            q_values.append(q_value)
        # q_values를 [batch_size, num_of_qubits, action_size]로 변환
        q_values = torch.stack(q_values, dim=0)  # [batch_size, num_of_qubits, action_size]
        return q_values  # [batch_size, num_of_qubits, action_size]


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        # Transition의 요소별로 묶어서 반환
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)


def train_dqn(X_train_transformed, data_size, action_size, Y_train, q_policy,
              q_target, optimizer, env, num_episodes, gamma, replay_buffer,
              batch_size, warm_up, target_interval):
    policy_losses = []  # 손실 값을 저장할 리스트
    total_step = 0

    for episode in range(num_episodes):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train_transformed,
                                               Y_train)
        state, _ = env.reset()

        done = False
        total_reward = 0

        while not done:
            total_step += 1
            # Epsilon-greedy 정책
            epsilon = 0.01  # 에피소드에 따라 감소시키는 방법을 사용할 수 있습니다.
            random_number = random.random()
            if random_number < epsilon:
                # 랜덤 행동 선택
                action = torch.randint(0, action_size, (data_size,), dtype=torch.long)
            else:
                # Q-값에 따라 행동 선택
                with torch.no_grad():
                    q_values = q_policy(state)
                    action = torch.max(q_values, dim=1)[1]

            # 환경에서 다음 상태, 보상 등 얻기
            next_state, reward, done = env.step(action.numpy(),
                                                X1_batch.numpy(),
                                                X2_batch.numpy(), Y_batch)
            total_reward += reward

            # 리플레이 버퍼에 저장
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state

            # 일정 시간마다 학습
            if len(replay_buffer) >= warm_up:  ##TODO
                # 미니배치 샘플링
                transitions = replay_buffer.sample(batch_size)
                batch = Transition(*transitions)

                # 배치 데이터 처리
                state_batch = torch.stack(batch.state)
                action_batch = torch.stack(batch.action)
                reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
                next_state_batch = torch.stack(batch.next_state)
                done_batch = torch.tensor(batch.done, dtype=torch.float32)

                # 현재 Q-값 계산
                q_values = q_policy(state_batch).permute(1, 0, 2)
                action_batch_expanded = action_batch.unsqueeze(-1)
                state_action_values = q_values.gather(2, action_batch_expanded).squeeze(-1)

                # 타깃 Q-값 계산
                with torch.no_grad():
                    next_q_values = q_target(next_state_batch).permute(1, 0, 2)
                    max_next_q_values = next_q_values.max(dim=2)[0]
                    target_values = reward_batch.unsqueeze(1) + gamma * max_next_q_values * (1 - done_batch.unsqueeze(1))

                # 손실 계산
                loss_fn = nn.MSELoss()
                loss = loss_fn(state_action_values, target_values)

                # 모델 최적화
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_policy.parameters(),
                                               max_norm=1.0)
                optimizer.step()

        # 타깃 네트워크 업데이트
        if total_step % target_interval == 0:  ## 이렇게 하면 warm up start 이후부터, interval 될때까지는 target network가 random parameterized 네트워크인긴 함. 그게 보통이라고 하네...
            q_target.load_state_dict(q_policy.state_dict())
        policy_losses.append(total_reward)

        print(f'Episode {episode + 1}/{num_episodes},Tt Reward: {total_reward}')

    return q_policy, policy_losses


# Function to generate action sequence
def generate_DQN_action_sequence(q_policy, batch_size, data_size,
                                 X_train_transformed, max_steps):
    env_eval = QASEnv(num_of_qubit=data_size, max_timesteps=max_steps,
                      batch_size=batch_size)
    state, _ = env_eval.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action_sequence = []

    for step in range(max_steps):
        with torch.no_grad():
            q_values = q_policy(state)  # [1, num_of_qubits, action_size]
            action = q_values.max(dim=2)[1]  # [1, num_of_qubits]

        action_sequence.append(action.squeeze(0).numpy())  # [num_of_qubits]

        # 다음 상태 얻기
        next_state = env_eval.step_eval(action.squeeze(0).numpy(),
                                        X_train_transformed)
        state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        if step % 3 == 0:
            print(f'{step + 1}/{max_steps} actions generated')

    return action_sequence
