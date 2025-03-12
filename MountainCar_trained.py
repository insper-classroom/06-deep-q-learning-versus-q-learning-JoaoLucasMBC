import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

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



env = gym.make('MountainCar-v0', render_mode='human').env
(state,_) = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
weights = torch.load('data/mountaincar_DeepQLearning.pth')
model = DQNModel(env.observation_space.shape[0], env.action_space.n)
model.load_state_dict(weights)
model.eval()

done = False
truncated = False
rewards = 0
steps = 0
max_steps = 1300

while (not done) and (not truncated) and (steps<max_steps):
    Q_values = model(state)
    action = torch.argmax(Q_values).item()
    state, reward, done, truncated, info = env.step(action)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    rewards += reward
    env.render()
    steps += 1

print(f'Score = {rewards}')
input('press a key...')