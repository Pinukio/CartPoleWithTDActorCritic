# 바닥부터 배우는 강화 학습 P.240 TD Actor-Critic 구현

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# HyperParameters
learning_rate = 0.0002
# discount factor
gamma = 0.98
n_rollout = 10