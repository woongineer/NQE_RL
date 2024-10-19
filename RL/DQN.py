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
        q_values = torch.stack(q_values,
                               dim=0)  # [batch_size, num_of_qubits, action_size]
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


def train_dqn(X_train_transformed, data_size, action_size, Y_train, policy_net,
              target_net, optimizer, env, num_episodes, gamma, replay_buffer,
              batch_size, target_update):
    policy_losses = []  # 손실 값을 저장할 리스트
    for episode in range(num_episodes):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train_transformed,
                                               Y_train)
        state, _ = env.reset()
        state = torch.tensor(state,
                             dtype=torch.float32)  # [batch_size, state_size * 4]
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # Epsilon-greedy 정책
            epsilon = 0.01  # 에피소드에 따라 감소시키는 방법을 사용할 수 있습니다.
            if random.random() < epsilon:
                # 랜덤 행동 선택
                action = torch.randint(0, action_size, (data_size,),
                                       dtype=torch.long)  # [batch_size, num_of_qubits]
            else:
                # Q-값에 따라 행동 선택
                with torch.no_grad():
                    q_values = policy_net(
                        state)  # [batch_size, num_of_qubits, action_size]
                    action = torch.max(q_values, dim=1)[
                        1]  # [batch_size, num_of_qubits]

            # 환경에서 다음 상태, 보상 등 얻기
            next_state, reward, done = env.step(action.numpy(),
                                                X1_batch.numpy(),
                                                X2_batch.numpy(), Y_batch)
            # next_state = torch.tensor(next_state, dtype=torch.float32)  # [batch_size, state_size * 4]
            total_reward += reward

            # 리플레이 버퍼에 저장
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            step_count += 1

            # 일정 시간마다 학습
            if len(replay_buffer) >= batch_size:
                # 미니배치 샘플링
                transitions = replay_buffer.sample(batch_size)
                batch = Transition(*transitions)

                # 배치 데이터 처리
                state_batch = torch.stack(
                    batch.state)  # [batch_size, state_size * 4]
                action_batch = torch.stack(
                    batch.action)  # [batch_size, num_of_qubits]
                reward_batch = torch.tensor(batch.reward,
                                            dtype=torch.float32)  # [batch_size]
                next_state_batch = torch.stack(
                    batch.next_state)  # [batch_size, state_size * 4]
                done_batch = torch.tensor(batch.done,
                                          dtype=torch.float32)  # [batch_size]

                # 현재 Q-값 계산
                q_values = policy_net(state_batch).permute(1, 0,
                                                           2)  # [batch_size, num_of_qubits, action_size]
                action_batch_expanded = action_batch.unsqueeze(
                    -1)  # [batch_size, num_of_qubits, 1]
                state_action_values = q_values.gather(2,
                                                      action_batch_expanded).squeeze(
                    -1)  # [batch_size, num_of_qubits]

                # 타깃 Q-값 계산
                with torch.no_grad():
                    next_q_values = target_net(next_state_batch).permute(1, 0,
                                                                         2)  # [batch_size, num_of_qubits, action_size]
                    max_next_q_values = next_q_values.max(dim=2)[
                        0]  # [batch_size, num_of_qubits]
                    target_values = reward_batch.unsqueeze(
                        1) + gamma * max_next_q_values * (
                                            1 - done_batch.unsqueeze(1))

                # 손실 계산
                loss_fn = nn.MSELoss()
                loss = loss_fn(state_action_values, target_values)

                # 모델 최적화
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(),
                                               max_norm=1.0)
                optimizer.step()

            # policy_losses.append(loss.item())

        # 타깃 네트워크 업데이트
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        policy_losses.append(total_reward)

        print(
            f'Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}')

    return policy_net, policy_losses


# Function to generate action sequence
def generate_DQN_action_sequence(policy_net, batch_size, data_size,
                                 X_train_transformed, max_steps):
    env_eval = QASEnv(num_of_qubit=data_size, max_timesteps=max_steps,
                      batch_size=batch_size)
    state, _ = env_eval.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(
        0)  # [1, state_size * 4]
    action_sequence = []

    for step in range(max_steps):
        with torch.no_grad():
            q_values = policy_net(state)  # [1, num_of_qubits, action_size]
            action = q_values.max(dim=2)[1]  # [1, num_of_qubits]

        action_sequence.append(action.squeeze(0).numpy())  # [num_of_qubits]

        # 다음 상태 얻기
        next_state = env_eval.step_eval(action.squeeze(0).numpy(),
                                        X_train_transformed)
        state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(
            0)  # [1, state_size * 4]

        if step % 3 == 0:
            print(f'{step + 1}/{max_steps} actions generated')

    return action_sequence
