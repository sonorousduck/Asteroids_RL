import random
import math
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Categorical

device = "cuda" if torch.cuda.is_available() else "cpu"

class PPO(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(observation_space, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, action_space.n)


        self.value = nn.Linear(in_features=512, out_features=1)
        self.pi_logits = nn.Linear(in_features=512, out_features=action_space.n)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))

        pi = Categorical(logits=self.pi_logits(x))
        value = self.value(x).reshape(-1)

        return pi, value

class Trainer:
    def __init__(self, updates, epochs, batch_size, value_loss_coef, entropy_bonus_coef, clip_range, lr):
        self.updates = updates
        self.epochs = epochs
        self.value_losss_coef = value_loss_coef
        self.entropy_bonus_coef = entropy_bonus_coef
        self.clip_range = clip_range
        self.lr = lr
        self.batch_size = batch_size
        
        self.model = PPO().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.ppo_loss = 



class PPOAgent:
    def __init__(self):
        pass