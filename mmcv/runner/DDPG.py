
import torch
from torch import nn, optim
import numpy as np
import os, sys
import pdb
import torchvision.models as models
import torch.nn.functional as F
from .replicate import patch_replication_callback
import collections
import random
batch_sizee = 64

class ReplayBuffer:
    def __init__(self, capacity):  # 经验池的最大容量
        # 创建一个队列，先进先出
        self.buffer = collections.deque(maxlen=capacity)
    # 在队列中添加数据
    def add(self, state, action, reward, next_state):
        # 以list类型保存
        self.buffer.append((state, action, reward, next_state))
    # 在队列中随机取样batch_size组数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 将数据集拆分开来
        state, action, reward, next_state= zip(*transitions)
        return state, action, reward, next_state
    # 测量当前时刻的队列长度
    def size(self):
        return len(self.buffer)

OPTIM_DICTS = {'sgd': optim.SGD,
               'adam': optim.Adam}

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

class Actor(nn.Module):
    def __init__(self, input_dim, num_fc,n_hiddens=16):
        super(Actor, self).__init__()
        # 只有一层隐含层的网络
        self.fc1 = nn.Linear(input_dim, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, num_fc)

    def forward(self, x):
        # pdb.set_trace()
        x = self.fc1(x.float())  # [b, states]==>[b, n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # 对batch中的每一行样本计算softmax，q值越大，概率越大
        x = F.softmax(x, dim=1)  # [b, n_actions]==>[b, n_actions]
        # pdb.set_trace()
        return x

class Critic(nn.Module):
    def __init__(self, input_dim, num_action,n_hiddens=16):
        super(Critic, self).__init__()
        # 只有一层隐含层的网络
        self.fc1 = nn.Linear(input_dim+num_action, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = self.fc1(x.float())  # [b, states]==>[b, n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class DDPG(nn.Module):
    def __init__(self, args, input_dim, num_action=2, use_cuda=True):
        '''
        args: optimizer types, lrs, momentum_weights, momentum_interval for actor and critic
        input_dim: the channel number for inputs
        d_dim: the output dim for the first convolution
        '''
        super(DDPG, self).__init__()
        self.args = args
        # pdb.set_trace()
        self.actor = Actor(input_dim, num_action)
        self.actor_ = Actor(input_dim, num_action)
        self.actor_optimizer = OPTIM_DICTS[self.args['actor_optimizer']](self.actor.parameters(), lr=self.args['actor_lr'], weight_decay=0.02)
        self.critic = Critic(input_dim, num_action)
        self.critic_ = Critic(input_dim, num_action)
        self.critic_optimizer = OPTIM_DICTS[self.args['critic_optimizer']](self.critic.parameters(), lr=self.args['critic_lr'], weight_decay=0.02)

        self.criterion = nn.MSELoss()

        hard_update(self.actor, self.actor_)
        hard_update(self.critic, self.critic_)
        self.use_cuda =use_cuda

        if self.args['cuda']:

            self.actor = torch.nn.DataParallel(self.actor)
            patch_replication_callback(self.actor)
            self.actor = self.actor.cuda()

            self.actor_ = torch.nn.DataParallel(self.actor_)
            patch_replication_callback(self.actor_)
            self.actor_ = self.actor_.cuda()

            self.critic = torch.nn.DataParallel(self.critic)
            patch_replication_callback(self.critic)
            self.critic = self.critic.cuda()

            self.critic_ = torch.nn.DataParallel(self.critic_)
            patch_replication_callback(self.critic_)
            self.critic_ = self.critic_.cuda()

        if self.args['rl_resume'] is not None:
            if not os.path.isfile(self.args['rl_resume']):
                raise RuntimeError("=> no checkpoint found at {}" .format(self.args['rl_resume']))
            ckpt = torch.load(self.args['rl_resume'])
            self.load(ckpt)
            print('{} loaded'.format(self.args['rl_resume']))

    def train(self):
        self.actor.train()
        self.critic.train()
        self.actor_.eval()
        self.critic_.eval()


    def update_policy(self, transition_dict, num_step, soft_update_interval):
        state_batch = torch.stack(transition_dict['states'])  # [b,n_states]
        action_batch = torch.stack(transition_dict['actions']).view(-1,1)  # [b,1]
        reward_batch = torch.stack(transition_dict['rewards']).view(-1,1)  # [b,1]
        next_state_batch = torch.stack(transition_dict['next_states'])  # [b,next_states]
             
        with torch.no_grad():
            next_q_values = self.critic_(next_state_batch, self.actor_(next_state_batch))
            target_q_batch = reward_batch + self.args['rl_discount'] * next_q_values.squeeze(-1)

        action_batch = torch.cat([1-action_batch,action_batch],dim=1)
        self.critic.zero_grad()
        q_batch = self.critic(state_batch, action_batch)
        # print(target_q_batch)
        # print(q_batch)
        value_loss = self.criterion(target_q_batch, q_batch)
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        if not num_step % soft_update_interval:
            self.update_target()

        return value_loss, policy_loss

    def update_target(self):
        soft_update(self.actor_, self.actor, self.args['rl_momentum'])
        soft_update(self.critic_, self.critic, self.args['rl_momentum'])

    def eval(self):
        self.actor.eval()
        self.actor_.eval()
        self.critic.eval()
        self.critic_.eval()

    def _actor_forward(self, x):
        with torch.no_grad():
            state = x.reshape(1,-1)
            probs = self.actor(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            action = action.item()
            return action
    
    def _actor_forward_(self, x):
        with torch.no_grad():
            state = x.reshape(1,-1)
            probs = self.actor_(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            action = action.item()
            return action

    def load(self, ckpt):
        if self.args['cuda']:
            self.actor.module.load_state_dict(ckpt['actor'])
            self.actor_.module.load_state_dict(ckpt['actor_'])
            self.actor.module.load_state_dict(ckpt['actor'])
            self.actor_.module.load_state_dict(ckpt['actor_'])
        else:
            self.actor.load_state_dict(ckpt['actor'])
            self.actor_.load_state_dict(ckpt['actor_'])
            self.actor.load_state_dict(ckpt['actor'])
            self.actor_.load_state_dict(ckpt['actor_'])
        if self.args['rl_ft']:
            self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
            self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])

    def save(self, target_name):
        if self.args['cuda']:
            save_target = {'actor': self.actor.module.state_dict(),
                           'actor_': self.actor_.module.state_dict(),
                           'critic': self.critic.module.state_dict(),
                           'critic_': self.critic_.module.state_dict(),
                           'actor_optimizer': self.actor_optimizer.state_dict(),
                           'critic_optimizer': self.critic_optimizer.state_dict()}
        else:
            save_target = {'actor': self.actor.state_dict(),
                           'actor_': self.actor_.state_dict(),
                           'critic': self.critic.state_dict(),
                           'critic_': self.critic_.state_dict(),
                           'actor_optimizer': self.actor_optimizer.state_dict(),
                           'critic_optimizer': self.critic_optimizer.state_dict()}
        torch.save(save_target, target_name)




