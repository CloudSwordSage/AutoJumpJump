# -*- coding: utf-8 -*-
# @Time    : 2024/7/24 00:26
# @Author  : chenfeng
# @Email   : zlf100518@163.com
# @FileName: play.py
# @LICENSE : MIT

import time
import os
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F

from env import AutoJumpEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f'    Using {device} device.')

actions = list(range(0, 2001, 10))

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 157 * 72, len(actions))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

env = AutoJumpEnv(hwnd=0x01150AA0, dpi=1, device=device)

policy_net = torch.load('./model/dqn-policy-whole.pth')
state = env.state().unsqueeze(0)
for t in count():
    action = policy_net(state).max(1)[1].view(1, 1)
    action = actions[action.item()]
    print(f'Seleced action: {action}ms')
    _, reward, done, score = env.step(action)
    time.sleep(4)
    reward = torch.tensor([reward], device=device)
    if not done:
        next_state = env.state().unsqueeze(0)
    else:
        next_state = None
    state = next_state
    if done:
        break