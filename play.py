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
import hwnd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'    Using {device} device.')

actions = list(range(0, 2001, 10))

class DQN(nn.Module):
    def __init__(self, c, h, w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        size = lambda x, y, z: int((x - y) / z + 1)
        oh = size(size(size(h, 8, 4), 4, 2), 3, 1)
        ow = size(size(size(w, 8, 4), 4, 2), 3, 1)
        linear_input_size = 64 * oh * ow
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, len(actions))


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

env = AutoJumpEnv(hwnd=hwnd.hwnd, dpi=1, device=device)

policy_net = torch.load('./model/dqn-policy-whole.pth')
for t in count():
    if env.end():
        print('End!')
        break
    state = env.state().unsqueeze(0)
    action = policy_net(state).max(1)[1].view(1, 1)
    action = actions[action.item()]
    print(f'Seleced action: {action}ms')
    env.step(action)
    time.sleep(4)