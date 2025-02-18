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
from .utils import get_configuration
import math

__all__ = ['FinalBlocksWorldEnv']


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
    
  def _get_n_stacks(self,mode):
    init_config   = get_configuration(self , mode=mode)
    n_stacks_init = 0
    for i in range(len(init_config)):
        if(init_config[i]==len(init_config)):
            n_stacks_init+=1
    return n_stacks_init

  def _get_free_space(self):
    space = self.max_n_stacks-self._get_n_stacks('initial')
    assert space>=0
    return space

  def _restart(self):
    redo = True
    while(redo):
        self.start_world = randomly_generate_world(
            self.nr_blocks, random_order=False)
        self.final_world = randomly_generate_world(
            self.nr_blocks, random_order=False)
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
        self.max_n_stacks = max(self._get_n_stacks('initial'), self._get_n_stacks('target'))
        if(self.max_n_stacks>math.ceil(self.nr_blocks*0.5)):
            redo=True
        else:
            redo=False
    self.max_n_stacks = math.ceil(self.nr_blocks*0.8)
    self.start_state = decorate(
        self._get_coordinates(self.start_world), self.nr_objects, 0)
    self.final_state = decorate(
        self._get_coordinates(self.final_world), self.nr_objects, 1)

    self.is_over = False
    self.cached_result = self._get_result()

  def _prepare_worlds(self):
    pass

  def _action(self, action):
    assert self.start_world is not None, 'you need to call restart() first'

    if self.is_over:
      return 0, True
    r, is_over = self.cached_result
    if is_over:
      self.is_over = True
      return r, is_over

    x, y = action
    assert 0 <= x <= self.nr_blocks and 0 <= y <= self.nr_blocks

    p = random.rand()
    self.prob_unchange = 0#0.05 #0.05
    self.prob_fall     = 0#0.1 #0.05
    
    if p >= self.prob_unchange:
      if p < self.prob_unchange + self.prob_fall:
        if(self.start_world.moveable(x,y)):
            y  = self.start_world.get_placeable(x)
      self.start_world.move(x, y, free_space = self._get_free_space())
      self.start_state = decorate(
          self._get_coordinates(self.start_world), self.nr_objects, 0)
    r, is_over = self._get_result()
    if is_over:
      self.is_over = True
    return r, is_over

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
    #coordinates = get_coordinates(world, absolute=not self.shape_only)
    coordinates = get_coordinates(world, absolute=not self.shape_only, free_space = self._get_free_space())
    if sort:
      if not self.shape_only:
        coordinates = decorate(coordinates, self.nr_objects, 0)
      coordinates = np.array(sorted(list(map(tuple, coordinates))))
    return coordinates