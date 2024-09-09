import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
action_dim = 1  # Number of continuous action dimensions

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable log standard deviation
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mean = self.fc2(x)
        std = torch.exp(self.log_std)  # Standard deviation is positive
        return mean, std

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, mean, std, log_prob in self.data[::-1]:
            R = r + gamma * R
            loss = -log_prob * R  # Policy gradient loss
            loss.backward()
        self.optimizer.step()
        self.data = []

def main():
    env = gym.make('Pendulum-v1')  # Use a continuous action environment
    pi = Policy()
    score = 0.0
    print_interval = 20

    for n_epi in range(20):
        s = env.reset()
        done = False

        while not done:  # Pendulum-v1 terminates when episode length exceeds a limit
            mean, std = pi(torch.from_numpy(s).float())
            dist = Normal(mean, std)
            a = dist.sample()
            log_prob = dist.log_prob(a).sum()  # Summing over action dimensions
            s_prime, r, done, _ = env.step(a.numpy())
            pi.put_data((r, mean, std, log_prob))
            s = s_prime
            score += r

        pi.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {}".format(n_epi,
                                                            score / print_interval))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()
