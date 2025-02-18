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
"""The environment class for blocks world tasks."""

import numpy as np

import jacinle.random as random
from jaclearn.rl.env import SimpleRLEnvBase

from .block import randomly_generate_world
from .represent import get_coordinates
from .represent import decorate
from .utils import Reward_Fetcher
from .utils import get_configuration
from jactorch.utils.meta import as_cuda
from jactorch.utils.meta import as_numpy
from jactorch.utils.meta import as_tensor
import torch
import torch.nn as nn
import torch.nn.functional as F



import math
__all__ = ['FinalBlocksWorldEnv']

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
    
    
class BlocksWorldEnv(SimpleRLEnvBase):
  """The Base BlocksWorld environment.

    Args:
        nr_blocks: The number of blocks.
        random_order: randomly permute the indexes of the blocks. This
          option prevents the models from memorizing the configurations.
        decorate: if True, the coordinates in the states will also include the
            world index (default: 0) and the block index (starting from 0).
        prob_unchange: The probability that an action is not effective.
        prob_fall: The probability that an action will make the object currently
            moving fall on the ground.
  """

  def __init__(self,
               nr_blocks,
               random_order=False,
               decorate=False,
               prob_unchange=0.0,
               prob_fall=0.0):
    super().__init__()
    self.nr_blocks = nr_blocks
    self.nr_objects = nr_blocks + 1
    self.random_order = random_order
    self.decorate = decorate
    self.prob_unchange = prob_unchange
    self.prob_fall = prob_fall

  def _restart(self):
    self.world = randomly_generate_world(
        self.nr_blocks, random_order=self.random_order)
    self._set_current_state(self._get_decorated_states())
    self.is_over = False
    self.cached_result = self._get_result()

  def _get_decorated_states(self, world_id=0):
    state = get_coordinates(self.world)
    if self.decorate:
      state = decorate(state, self.nr_objects, world_id)
    return state


class FinalBlocksWorldEnv(BlocksWorldEnv):
  """The BlocksWorld environment for the final task."""

  def __init__(self,
               nr_blocks,
               random_order=False,
               shape_only=False,
               fix_ground=False,
               prob_unchange=0.0,
               prob_fall=0.0):
    super().__init__(nr_blocks, random_order, True, prob_unchange, prob_fall)
    self.shape_only = shape_only
    self.fix_ground = fix_ground
    
    self.rf = None
    self.run_mode=None
    self.nActions= 0
    self.max_n_stacks = self.nr_blocks#3#self.nr_blocks #3
    self.max_len = 2*get_length(self.nr_blocks)#8*self.nr_blocks #2*phi(self.max_n_stacks,self.nr_blocks)#4*self.nr_blocks#2*phi(4,self.nr_blocks)#4*self.nr_blocks
    self.threshold = 100
    self.baseline_lengths = None#{}
    self.baseline_candidates = {}
    self.rs = []
    
  def _restart(self):
    redo = True
    while(redo):
        redo=False
        self.start_world = randomly_generate_world(
            self.nr_blocks, random_order=False)
        self.final_world = randomly_generate_world(
            self.nr_blocks, random_order=False)
        self.start_world.max_n_stacks=self.max_n_stacks
        self.final_world.max_n_stacks=self.max_n_stacks
        self.world = self.start_world
        if self.random_order:
          n = self.world.size
          # Ground is fixed as index 0 if fix_ground is True
          ground_ind = 0 if self.fix_ground else random.randint(n)

          def get_order():
            raw_order = random.permutation(n - 1)
            order = []
            for i in range(n - 1):
              if i == ground_ind:
                order.append(0)
              order.append(raw_order[i] + 1)
            if ground_ind == n - 1:
              order.append(0)
            return order
          self.start_world.blocks.set_random_order(get_order())
          self.final_world.blocks.set_random_order(get_order())
        self._prepare_worlds()
        '''
        if(max(self._get_n_stacks('initial'), self._get_n_stacks('target')) > self.max_n_stacks):#self.max_n_stacks):
            redo=True
        else:
            redo=False
        '''
        
        
    self.target_n_stacks = self._get_n_stacks('target')
    self.start_state = decorate(
        self._get_coordinates(self.start_world), self.nr_objects, 0)
    self.final_state = decorate(
        self._get_coordinates(self.final_world), self.nr_objects, 1)

    self.is_over = False
    self.cached_result = self._get_result()
    
    self.rf = None
    self.run_mode=None
    self.nActions= 0
    self.plan_size     = None#int(max((self.threshold/100)*self.max_len   + (1-self.threshold/100)*self.nr_blocks, self.nr_blocks)) #math.ceil(phi(self.max_n_stacks,self.nr_blocks)/2)
    self.interval_size = 1#None#int(max((self.threshold/100)*self.nr_blocks + (1-self.threshold/100)*1, 1))
    self.inconsistency_found = False
    #self.max_len = 8*self.nr_blocks #(phi(4,self.nr_blocks)*2)
    self.action_distribution = None
    self.model = None
    self.norm = None
    self.rs = []
    self.es = []
  def set_threshold(self, threshold):
    self.threshold = threshold
    #self.plan_size     = int(max((self.threshold/100)*self.max_len   + (1-self.threshold/100)*self.nr_blocks, self.nr_blocks)) #math.ceil(phi(self.max_n_stacks,self.nr_blocks)/2)
    #self.interval_size = int(max((self.threshold/100)*self.nr_blocks + (1-self.threshold/100)*1, 1))
  def set_baseline_lengths(self, baseline_lengths):
    self.baseline_lengths = baseline_lengths
    
    #print('self.plan_size:',self.plan_size)
  def get_string_representation(self):
    r_initial = '-'.join([str(i) for i in get_configuration(self , mode='initial')])
    r_target  = '-'.join([str(i) for i in get_configuration(self , mode='target')])
    r = r_initial+'#'+r_target
    return r
  def get_plan_size(self):
    #r = self.get_string_representation()
    #base_l = 2*self.nr_blocks#self.max_len - self.nActions
    #if(r in self.baseline_lengths.keys()):
    #    base_l = self.baseline_lengths[self.get_string_representation()]
    return self.plan_size
    #return base_l
  def _get_n_stacks(self,mode):
    init_config   = get_configuration(self , mode=mode)
    n_stacks_init = 0
    for i in range(len(init_config)):
        if(init_config[i]==len(init_config)):
            n_stacks_init+=1
    return n_stacks_init

  def _get_free_space(self):
    #return 1
    space = self.max_n_stacks-self._get_n_stacks('initial')
    assert space>=0
    return space

  def _prepare_worlds(self):
    pass

  def _create_rf(self):
    plan_size = self.get_plan_size()
    self.baseline_candidates[self.get_string_representation()] = self.nActions
    if((not self.rf==None) and self.rf.min_cost > self.max_len):
        return self.rf, (self.rf.inconsistent==1)
    rf = Reward_Fetcher(self, length_multiplier=self.nActions, gamma=1, plan_size_try=plan_size, free_space=self._get_free_space(), max_stacks=self.max_n_stacks, target_stacks = self.target_n_stacks, max_len = self.max_len, threshold=100, baseline_lengths=self.baseline_lengths)
    inconsistent_over   = False
    while(rf.inconsistent==1):
        plan_size += self.interval_size
        if(rf.min_cost>self.max_len): #and (rf.inconsistent==1)):
            inconsistent_over = True
            break
        else:
            #print('inconsistency detected')
            rf = Reward_Fetcher(self, length_multiplier=self.nActions, gamma=1, plan_size_try=plan_size, free_space=self._get_free_space(), max_stacks=self.max_n_stacks, target_stacks = self.target_n_stacks, max_len = self.max_len, threshold=100, baseline_lengths=self.baseline_lengths)
    return rf, inconsistent_over

  def _action(self, action):
    assert self.start_world is not None, 'you need to call restart() first'
    
    if((self.rf == None) and (self.run_mode=='train')):
        self.rf, inconsistent_over = self._create_rf()
    
    if self.is_over:
      return 0, True
    r, is_over = self.cached_result
    if is_over:
      self.is_over = True
      return 0*r, is_over
    
    #if((self.rf == None) and (self.run_mode=='train')):
    #    self.rf, inconsistent_over = self._create_rf()
        #print('tail changed to (1)',self.plan_tail)
        #print('initial plan_tail:',self.plan_tail)
        #self.rf = Reward_Fetcher(self, length_multiplier=self.nActions, gamma=1, plan_size_try=self.plan_size)


    #print('before:',get_configuration(self,'initial'), 'target:',get_configuration(self,'target'))
    x, y = action
    #self.baseline_candidates[self.get_string_representation()] = self.nActions
    
    assert 0 <= x <= self.nr_blocks and 0 <= y <= self.nr_blocks
    movable = self.start_world.moveable_2(x,y)
    ground_change = self.start_world.ground_change(x,y)
    p = random.rand()
    self.prob_unchange = 0#0.05 #0.05
    self.prob_fall     = 0#0.1 #0.05
    if p >= self.prob_unchange:
      if p < self.prob_unchange + self.prob_fall:
        if(movable):
            y  = self.start_world.get_placeable(x)
        #y = self.start_world.blocks.inv_index(0)  # fall to ground
        
      self.start_world.move(x, y)#, free_space = self._get_free_space())
      if(self.run_mode=='train'):
          self.initiate_distribution()
      self.nActions +=1
      self.start_state = decorate(
          self._get_coordinates(self.start_world), self.nr_objects, 0)
    else:
        self.nActions += 1
        x,y = 0,0
        movable = False

    #print('after:',get_configuration(self,'initial'))
    
    r, is_over = self._get_result()
    r_exp, r_min = 0, 0
    '''
    if(ground_change==0):
        assert(fs_after==fs_before)
    elif(ground_change==1):
        assert(fs_after==fs_before+1)
    elif(ground_change==2):
        assert(fs_after==fs_before-1)
    else:
        assert(False)
    '''
    lm=4
    inconsistent_over=False
    if(self.run_mode=='train'):
        r_exp, r_min = self.rf.move_raw(x,y,is_over,movable,ground_change)
        prev_rf = self.rf
        inconsistent_over = (self.rf.min_cost > self.max_len) and (self.rf.inconsistent==1)
        if(not inconsistent_over and (self.rf.inconsistent==1)):#(self.rf.inconsistent==1 and (not self.rf.exp_cost == 4*self.nr_blocks)):
            #print('create_new!')
            #print(self.rf.inconsistent==1)
            rf, inconsistent_over = self._create_rf()
            self.rf = rf
            r_exp = (self.rf.exp_cost) - prev_rf.prev_exp_cost

    if(inconsistent_over):
        self.inconsistency_found = True
        
    is_over = is_over or self.inconsistency_found
    if is_over:
      self.is_over = True
    if(self.run_mode=='train'):
        r_exp = -(r_exp)#self.norm
    #self.rs+=[r_exp]
    ws = [1, 1]
    
    #if(is_over_):
        #print(r_exp)
    
    #if(self.inconsistency_found and r==1):
    #    print('inconsistency and success (!)')
    #return (-1+0*r), is_over
    return (r_exp+0*r), is_over

  def _get_current_state(self):
    assert self.start_world is not None, 'Should call restart() first.'
    return np.vstack([self.start_state, self.final_state])

  def _get_result(self):
    sorted_start_state = self._get_coordinates(self.start_world, sort=True)
    sorted_final_state = self._get_coordinates(self.final_world, sort=True)
    if (sorted_start_state == sorted_final_state).all():
      return 1, True
    else:
      return 0, False

  def _get_coordinates(self, world, sort=False):
    # If shape_only=True, only the shape of the blocks need to be the same.
    # If shape_only=False, the index of the blocks should also match.
    coordinates = get_coordinates(world, absolute=not self.shape_only, free_space = self._get_free_space())
    if sort:
      if not self.shape_only:
        coordinates = decorate(coordinates, self.nr_objects, 0)
      coordinates = np.array(sorted(list(map(tuple, coordinates))))
    return coordinates

  def initiate_distribution(self):
    '''
    state = self._get_current_state()
    feed_dict = dict(states=np.array([state]))
    feed_dict = as_tensor(feed_dict)
    feed_dict = as_cuda(feed_dict)
    self.model.training=False
    with torch.set_grad_enabled(False):
      output_dict = self.model(feed_dict)
    policy = output_dict['policy']
    p = as_numpy(policy.data[0])
    action_distribution = []
    for i in range(len(p)):
        action_distribution += [(self.mapping[i], p[i])]
    self.action_distribution = action_distribution
    '''
    self.action_distribution = []
        