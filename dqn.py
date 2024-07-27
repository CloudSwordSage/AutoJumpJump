# -*- coding: utf-8 -*-
# @Time    : 2024/7/24 00:26
# @Author  : chenfeng
# @Email   : zlf100518@163.com
# @FileName: dqn.py
# @LICENSE : MIT

import math
import random
import time
import os
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from env import AutoJumpEnv
import hwnd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.system('cls')

print(f'    Using {device} device.')

actions = list(range(0, 2001, 10))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
c, h, w = env.state().shape

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 0.00025

print(f'''    BATCH_SIZE: {BATCH_SIZE}
    GAMMA: {GAMMA}
    EPS_START: {EPS_START}
    EPS_END: {EPS_END}
    EPS_DECAY: {EPS_DECAY}
    TAU: {TAU}
    LR: {LR}\n''')

policy_net_path = './model/dqn-policy.pth'
target_net_path = './model/dqn-target.pth'

if os.path.exists('./model/dqn-policy.pth'):
    print('    Loading policy network...')
    policy_net = torch.load(policy_net_path)
else:
    print('    Creating policy network...')
    policy_net = DQN(c, h, w).to(device)

if os.path.exists('./model/dqn-target.pth'):
    print('    Loading target network...')
    target_net = torch.load(target_net_path)
else:
    print('    Creating target network...')
    target_net = DQN(c, h, w).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
print('    Creating memory...')
memory = ReplayMemory(10000)
print(f'    memory size: 10000')

steps_done = 1000 if os.path.exists('./model/dqn-policy.pth') and \
    os.path.exists('./model/dqn-target.pth') else 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        del sample, eps_threshold
        torch.cuda.empty_cache()
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1), 'policy'
    else:
        del sample, eps_threshold
        torch.cuda.empty_cache()
        return torch.tensor([[random.randint(0, len(actions) - 1)]], device=device, dtype=torch.long), 'random'
episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return False
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    del transitions, batch, non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch,\
    state_action_values, next_state_values, expected_state_action_values, loss, param
    torch.cuda.empty_cache()
    return True

num_episodes = 600
print(f'    num_episodes: {num_episodes}')
print('-'*110)
episode_tar = tqdm(total=num_episodes, desc='Training', unit='episodes', leave=False)
score_list = []
avg_score_list = []
is_opt = False

for i_episode in range(num_episodes):
    state = env.state().unsqueeze(0)
    for t in count():
        action, source = select_action(state)
        _, reward, done, score = env.step(actions[action.item()])
        text = f'source: {source} | episode: {i_episode: <3} | step: {t+1: <3} | action: {actions[action.item()]: <5}'
        text += f' | reward: {reward: <5} | score: {score: <5} | optimize model: {is_opt}'
        tqdm.write(text)
        time.sleep(4)
        if (not done) or reward > 0:
            next_state = env.state().unsqueeze(0)
        else:
            next_state = None
        reward = torch.tensor([reward], device=device)
        memory.push(state, action, next_state, reward)
        state = next_state
        is_opt = optimize_model()
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        if done:
            episode_durations.append(t + 1)
            break
    del target_net_state_dict, policy_net_state_dict
    torch.cuda.empty_cache()
    score_list.append(score)
    avg_score = sum(score_list[-10:])/len(score_list[-10:])
    avg_score_list.append(avg_score)
    tqdm.write('-'*110)
    episode_tar.set_postfix(Duration=t+1, score=score)
    episode_tar.update()
    if i_episode % 50 == 49:
        txt = f'{"Saving model...":=^110}'
        tqdm.write(txt)
        torch.save(policy_net, policy_net_path)
        torch.save(target_net, target_net_path)
        txt = f'{"Saved model":=^110}'
        tqdm.write(txt)
print('Complete')
torch.save(policy_net, './model/dqn-policy-whole.pth')
episode_tar.close()
plt.figure(1)
plt.title('Result')
plt.xlabel('Episode')
plt.ylabel('Scores')
plt.plot(score_list, label='score', color='blue')
plt.plot(avg_score_list, label='average score', color='red')
plt.legend()
plt.show()