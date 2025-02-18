#! /usr/bin/env python3
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implement REINFORCE loss."""

import torch.nn as nn
import numpy as np
__all__ = ['REINFORCELoss']

def phi(p,N):
    if(p==3):
        return (2**N)-1
    if(N==1):
        return 1
    min_cand = None
    for l in range(1,N):
        res = 2*phi(p,l)+phi(p-1,N-l)
        if(min_cand == None or res<min_cand):
            min_cand = res
    return min_cand

def get_length(nr_blocks):
    if(nr_blocks>=5):
        return 4*nr_blocks
    else:
        return 4*nr_blocks

class REINFORCELoss(nn.Module):
  """Implement the loss function for REINFORCE algorithm."""

  def __init__(self, entropy_beta=None):
    super().__init__()
    self.nll = nn.NLLLoss(reduce=False)
    self.entropy_beta = entropy_beta
    #self.trainer = None
    
  def forward_redundant(self, policy, action, discount_reward, succ, number,index=None):
      #print('index:',index,'shape:',policy[index].shape)
      nll = self.nll(policy[index], action[index])
      monitors = dict()
      #print(nll)
      return nll, monitors
  
  def forward(self, policy, action, discount_reward, succ, number, entropy_beta=None,optim=None,redundancy=False,index=None):
    monitors = dict()
    entropy = -(policy * policy.log()).sum(dim=1).mean()
    nll = self.nll(policy, action)
    loss = (nll * discount_reward).mean()
    if entropy_beta is None:
      entropy_beta = self.entropy_beta
    if entropy_beta is not None:
      monitors['reinforce_loss'] = loss
      monitors['entropy_loss'] = -entropy * entropy_beta
      loss -= entropy * entropy_beta
    monitors['entropy'] = entropy
    return loss, monitors
   
