import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from DeepQLearningTorch import DeepQLearning
import csv


class DQNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # First Dense layer
        self.fc2 = nn.Linear(256, output_dim)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # No activation on output (equivalent to linear activation)

    def save(self, filename):
        torch.save(self.state_dict(), filename)


if __name__ == '__main__':
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_dec = 0.995
    episodes = 200
    batch_size = 64
    memory = deque(maxlen=10000) #talvez usar uma memoria mais curta
    max_steps = 1300
    alpha = 0.001


    env = gym.make('MountainCar-v0')
    #env.seed(0)
    np.random.seed(0)

    print('State space: ', env.observation_space)
    print('Action space: ', env.action_space)

    # Instantiate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = DQNModel(env.observation_space.shape[0], env.action_space.n).to(device)
    optimizer = optim.Adam(model.parameters(), lr=alpha)
    loss_fn = nn.MSELoss()

    DQN = DeepQLearning(env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model, max_steps, device, loss_fn, optimizer)
    rewards = DQN.train()


    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('# Rewards')
    plt.title('# Rewards vs Episodes')
    plt.savefig("results/mountaincar_DeepQLearning.jpg")     
    plt.close()

    with open('results/mountaincar_DeepQLearning_rewards.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        episode=0
        for reward in rewards:
            writer.writerow([episode,reward])
            episode+=1

    model.save('data/mountaincar_DeepQLearning.pth')