import collections
import copy
import functools
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import jacinle.random as random
import jacinle.io as io

import pickle

from difflogic.cli import format_args
from difflogic.dataset.utils import ValidActionDataset
from difflogic.envs.blocksworld import make as make_env
from difflogic.nn.baselines import MemoryNet
from difflogic.nn.neural_logic import InputTransform
from difflogic.nn.neural_logic import LogicInference
from difflogic.nn.neural_logic import LogicMachine
from difflogic.nn.neural_logic import LogitsInference
from difflogic.nn.neural_logic.modules._utils import meshgrid_exclude_self
from difflogic.nn.rl.reinforce import REINFORCELoss
from difflogic.train import MiningTrainerBase

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger
from jacinle.logging import set_output_file
from jacinle.utils.container import GView
from jacinle.utils.meter import GroupMeters
from jactorch.optim.accum_grad import AccumGrad
from jactorch.optim.quickaccess import get_optimizer
from jactorch.utils.meta import as_cuda
from jactorch.utils.meta import as_numpy
from jactorch.utils.meta import as_tensor
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class Model_NLM(nn.Module):
  """The model for blocks world tasks."""

  def __init__(self, args):
    super().__init__() 
    self.training=True
    self.args=args
    #print(self.args.use_gpu)
    # The 4 dimensions are: world_id, block_id, coord_x, coord_y
    input_dim = 4+2
    self.transform = InputTransform('cmp', exclude_self=False)

    # current_dim = 4 * 3 = 12
    current_dim = transformed_dim = self.transform.get_output_dim(input_dim)
    self.feature_axis = 1 if args.concat_worlds else 2

    input_dims = [0 for _ in range(args.nlm_breadth + 1)]
    input_dims[2] = current_dim
    self.features = LogicMachine.from_args(input_dims, args.nlm_attributes, args, prefix='nlm')
    current_dim = self.features.output_dims[self.feature_axis]
    self.final_transform = InputTransform('concat', exclude_self=False)
    if args.concat_worlds:
      current_dim = (self.final_transform.get_output_dim(current_dim) +
                     transformed_dim) * 2
    self.pred = LogitsInference(current_dim, 1, [])
    self.loss = REINFORCELoss()
    
  def get_binary_relations(self, states, depth=None):
    args = self.args
    """get binary relations given states, up to certain depth."""
    # total = 2 * the number of objects in each world
    total = states.size()[1]
    f = self.transform(states)
    if args.model == 'memnet':
      f = self.feature(f)
    else:
      inp = [None for i in range(args.nlm_breadth + 1)]
      inp[2] = f
      features = self.features(inp, depth=depth)
      f = features[self.feature_axis]

    assert total % 2 == 0
    nr_objects = total // 2
    if args.concat_worlds:
      # To concat the properties of blocks with the same id in both world.
      f = torch.cat([f[:, :nr_objects], f[:, nr_objects:]], dim=-1)
      states = torch.cat([states[:, :nr_objects], states[:, nr_objects:]],
                         dim=-1)
      transformed_input = self.transform(states)
      # And perform a 'concat' transform to binary representation (relations).
      f = torch.cat([self.final_transform(f), transformed_input], dim=-1)
    else:
      f = f[:, :nr_objects, :nr_objects].contiguous()

    f = meshgrid_exclude_self(f)
    return f

  def forward(self, states):
    if (type(states) == torch.Tensor):
        ...
    else:
        states = torch.from_numpy(states).unsqueeze(0).cuda()
    #states = states.cpu()
    f = self.get_binary_relations(states)
    logits = self.pred(f).squeeze(dim=-1).view(states.size(0), -1)
    policy = F.softmax(logits, dim=-1).clamp(min=1e-20)
    return policy
  
    
