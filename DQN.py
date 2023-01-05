import random
import math
from collections import deque
import numpy as np
import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(observation_space, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, action_space.n)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        # x = F.relu(x)

        return x


class DQNAgent:
    def __init__(self, observation_space, action_space, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.975, alpha=0.01, alpha_decay=0.01, batch_size=64, lr=1e-4):
        self.memory = deque(maxlen=100_000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        self.lr = lr
        self.action_space = action_space
        self.observation_space = observation_space
        self.dqn = DQN(self.observation_space, self.action_space)
        self.dqn.cuda()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr)
    
    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def preprocess_state(self, state):
        return torch.tensor(np.reshape(state, [1, self.observation_space]), dtype=torch.float32).cuda()
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                return torch.argmax(self.dqn(state)).cpu().detach().numpy()
    
    def remember(self, state, action, reward, next_state, terminal, truncated):
        reward = torch.tensor(reward).cuda()
        self.memory.append((state, action, reward, next_state, terminal, truncated))

    def replay(self):
        if len(self.memory) < self.batch_size: 
            return

        y_batch = []
        y_target_batch = []

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, terminal, truncated in minibatch:
            y = self.dqn(state)
            y_target = y.clone().cpu().detach()
            with torch.no_grad():
                y_target[0][action] = reward if terminal or truncated else reward + self.gamma * torch.max(self.dqn(next_state)[0])
            y_batch.append(y[0])
            y_target_batch.append(y_target[0])
        
        y_batch = torch.cat(y_batch).cuda()
        y_target_batch = torch.cat(y_target_batch).cuda()

        self.optimizer.zero_grad()
        loss = self.criterion(y_batch, y_target_batch)
        loss.backward()
        self.optimizer.step()

        self.decay_epsilon()

    def save_model(self):
        torch.save(self.dqn.state_dict(), 'dqn_model.pth')
    
    def load_model(self):
        self.dqn.load_state_dict(torch.load('dqn_model.pth'))
    
    def eval_model(self):
        self.dqn.eval()
