import random
import math
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from actor_critic import *
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

class DDPGAgent:
    def __init__(self, env, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_mem_size=50_000):
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.hidden_size=hidden_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.actor = Actor(self.observation_space, self.hidden_size, self.action_space)
        self.actor_target = Actor(self.observation_space, self.hidden_size, self.action_space)
        self.critic = Critic(self.observation_space + self.action_space, hidden_size, self.action_space)
        self.critic_target = Critic(self.observation_space + self.action_space, hidden_size, self.action_space)

        with torch.no_grad():

        # Set target to be the same as actor (avoids differences in random initialization most likely)
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.copy_(param.data)

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)


        # Training
        self.memory = Memory(max_mem_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = self.actor(state)
        action = action.cpu().detach().numpy()[0, 0]
        return action
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        else:
           self.get_action(state)
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)

        q_vals = self.critic(states, actions.detach())
        next_actions = self.actor_target(next_states)
        next_q = self.critic_target(next_states, next_actions.detach())
        q_prime = rewards + self.gamma * next_q

        critic_loss = self.critic_criterion(q_vals, q_prime)
        policy_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
        # update target networks 
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    
    def decay(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)