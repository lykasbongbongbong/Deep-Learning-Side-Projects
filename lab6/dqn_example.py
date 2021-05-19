'''DLP DQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tqdm as tqdm

class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=32):
        super(Net,self).__init__()
        ## TODO: OK ##
        self.net = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=action_dim)
        )

    def forward(self, x):
        ## TODO ##
        out = self.net(x)
        return out
        # raise NotImplementedError


class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        ## TODO ##
        self._optimizer = optim.Adam(self._behavior_net.parameters(), lr=args.lr)
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        ## TODO ##
        if np.random.uniform() < epsilon:
            # < epsilon : 從action_space 隨機 sample
            action = action_space.sample()
        else:
            with torch.no_grad():
                # 挑最大qvalue的action
                current_state = torch.from_numpy(state).view(1,8).to(self.device)
                all_qvalues = self._behavior_net(current_state)
                action = torch.argmax(all_qvalues).item()

        return action 



    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, [action], [reward / 10], next_state,
                            [int(done)])

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        ## TODO ##

        # 拿A當前的state 用predict出來的action 算一次qvalue
        
        q_value = self._behavior_net(state)#橫的取，取對應action的value
        q_value = torch.gather(q_value, 1, action.long())
      
        with torch.no_grad():
            next_qvalue = self._target_net(next_state)
            max_next_qvalue = next_qvalue.max(dim=1)[0]  # 1*nA
            max_next_qvalue = max_next_qvalue.reshape(-1 ,1)
            q_target = reward + gamma * max_next_qvalue * (1-done)
            
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)


        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        ## TODO ##
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        # raise NotImplementedError

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])


def train(args, env, agent, writer):

    print('Start Training')
    action_space = env.action_space
    total_steps = 0  #為了記錄當前走的第幾步，因為要先存一些記憶，當記憶中有東西才開始學
    epsilon = 1.
    ewma_reward = 0

    for episode in tqdm.tqdm(range(args.episode)):
        total_reward = 0

        state = env.reset()  # initial observation 初始狀態
        epsilon = max(epsilon * args.eps_decay, args.eps_min)

        for t in itertools.count(start=1):  # itertools會一直做下去到break為止 (while(True))
            # select action
            if total_steps < args.warmup:
                # 因為在>warmup的step才會開始學，所以這邊先直接用sample的
                ###### Actions
                # 1: No-op
                # 2: fire left engine
                # 3: fire main engine
                # 4: fire right engine
                ######
                action = action_space.sample()  # ex. 1
            else:
                # 根據觀測值選一個action (forward propagation 得到q值來選)
                action = agent.select_action(state, epsilon, action_space) 
            # execute action: take action 然後得到下一個state 和 take action之後的reward (要不要terminate)
            next_state, reward, done, _ = env.step(action)
            # store transition: 存現在這一步的state / 會得到的reward / take action 之後的state 跟要不要結束
            agent.append(state, action, reward, next_state, done)

            # 終於看懂了我的天!!!!: 要當step大於warmup之後才會開始做update (就是上面講的記憶有東西之後才開始學), 在小於warmup之前的step都跳過
            if total_steps >= args.warmup:
                agent.update(total_steps)

            # 把當前的state更新成next_state
            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_steps, episode, t, total_reward, ewma_reward,
                            epsilon))
                break
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    action_space = env.action_space
    epsilon = args.test_epsilon
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        ## TODO ##
        for t in itertools.count(start=1):
            action = agent.select_action(state, epsilon, action_space)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward 
            if done: 
                print(f"Total Reward: {total_reward}")
                rewards.append(total_reward)
                break
            
    print('Average Reward', np.mean(rewards))
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='dqn.pth')
    parser.add_argument('--logdir', default='log/dqn')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=2400, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=.0005, type=float)
    parser.add_argument('--eps_decay', default=.995, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=1000, type=int) 
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    parser.add_argument('--test_epsilon', default=.001, type=float)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLander-v2')
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
