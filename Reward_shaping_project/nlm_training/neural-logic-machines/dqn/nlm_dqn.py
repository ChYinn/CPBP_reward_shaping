import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dqn.memory.buffer import PrioritizedReplayBuffer
from dqn.nlm_val import Model_DQN

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

    
class DQN(nn.Module):

    def __init__(self, args):
        super(DQN, self).__init__()
        args.depth = 7
        self.model = Model_DQN(args)
        self.target_model = Model_DQN(args)
        
        self.model.cuda()
        self.target_model.cuda()
        self.memory = None
        self.mem_shape   = None
        self.action_size = None
        self.gamma = 1.0
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.tau = 0.001
        
        
    def add(self, state, action, n_actions, reward, next_state, done, expected):
        self.init_memory(state.shape, n_actions)
        self.memory.add(state, action, reward, next_state, int(done), expected)
        
    def init_memory(self, mem_shape, action_size):
        if(not (self.mem_shape==mem_shape or self.action_size == action_size)):
            self.mem_shape   =  mem_shape
            self.action_size =  action_size
            #print((mem_shape[0], mem_shape[1]))
            self.memory = PrioritizedReplayBuffer(mem_shape[0], mem_shape[1], action_size, 10000)# ReplayMemory(10000)
        else:
            ...
    def soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)

    def act(self, state):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float).to(device())
            action = torch.argmax(self.model(state)).cpu().numpy().item()
        return action

    def update(self, batch, weights=None):
        states, actions, rewards, next_states, dones, expecteds = batch
        loss = 0
        td_error = []
        for i in range(states.shape[0]):
            state, action, reward, next_state, done, expected = states[i], actions[i], rewards[i], next_states[i], dones[i], expecteds[i]
            Q_next = self.target_model(next_state).max(dim=1).values
            #print('Q_next: ',Q_next)
            Q_target = reward + self.gamma * (1 - done) * (Q_next[0]*0.5 + expected*0.5)
            action = action.to(torch.long)[0]
            #print('action:',action)
            Q = self.model(state)[0][action]
            #print('Q: ',Q)
            assert Q.shape == Q_target.shape, f"{Q.shape}, {Q_target.shape}"

            if weights is None:
                weights = torch.ones_like(Q)
            weights=weights.to(device='cuda')
            #print(weights.device)
            td_error += [torch.abs(Q - Q_target).detach().unsqueeze(0)]
            loss += torch.mean((Q - Q_target)**2 * weights)
        loss /= states.shape[0]
        td_error = torch.stack(td_error)
        td_error = td_error.squeeze()
        #print(td_error)
        #print(td_error.shape)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.soft_update(self.target_model, self.model)
        #print(loss.item())
        return loss.item(), td_error
    def update_model(self):
        buffer = self.memory
        model  = self.model
        batch, weights, tree_idxs = buffer.sample(min(8, buffer.real_size))
        loss, td_error = self.update(batch, weights=weights)
        buffer.update_priorities(tree_idxs, td_error.cpu().numpy())
        
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        self.model.training=True
        with torch.set_grad_enabled(True):
          output_dict = self.model(x)
        return output_dict 