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
"""The script for blocks world experiments."""

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
from dqn.nlm_dqn import DQN

import math 
TASKS = ['final']

parser = JacArgumentParser()

parser.add_argument(
    '--model',
    default='nlm',
    choices=['nlm', 'memnet'],
    help='model choices, nlm: Neural Logic Machine, memnet: Memory Networks')

# NLM parameters, works when model is 'nlm'.
nlm_group = parser.add_argument_group ('Neural Logic Machines')
#CHANGED-PARAM
LogicMachine.make_nlm_parser(
    nlm_group, {
        'depth': 7,
        'breadth': 2,
        'residual': True,
        'exclude_self': True,
        'logic_hidden_dim': []
    },
    prefix='nlm')
nlm_group.add_argument(
    '--nlm-attributes',
    type=int,
    default=8,
    metavar='N',
    help='number of output attributes in each group of each layer of the LogicMachine'
)

# MemNN parameters, works when model is 'memnet'.
memnet_group = parser.add_argument_group('Memory Networks')
MemoryNet.make_memnet_parser(memnet_group, {}, prefix='memnet')

parser.add_argument(
    '--task', required=True, choices=TASKS, help='tasks choices')

method_group = parser.add_argument_group('Method')
method_group.add_argument(
    '--no-concat-worlds',
    action='store_true',
    help='concat the features of objects of same id among two worlds accordingly'
)
method_group.add_argument(
    '--pred-depth',
    type=int,
    default=None,
    metavar='N',
    help='the depth of nlm used for prediction task')

#CHANGED-PARAM
method_group.add_argument(
    '--pred-weight',
    type=float,
    default=0.1,
    metavar='F',
    help='the linear scaling factor for prediction task')

MiningTrainerBase.make_trainer_parser(
    parser, {
        'epochs': 500,
        'epoch_size': 100,
        'test_epoch_size': 1000,
        'test_number_begin': 10,
        'test_number_step': 10,
        'test_number_end': 50,
        'curriculum_start': 2,
        'curriculum_step': 1,
        'curriculum_graduate': 12, #CHANGED-PARAM
        'curriculum_thresh_relax': 0.005,
        'sample_array_capacity': 3,#CHANGED-PARAM
        'enable_mining': True,
        'mining_interval': 10,
        'mining_epoch_size': 3000,
        'mining_dataset_size': 300,#300,
        'inherit_neg_data': True,
        'prob_pos_data': 0.6
    })

train_group = parser.add_argument_group('Train')
train_group.add_argument('--seed', type=int, default=None, metavar='SEED')
train_group.add_argument(
    '--use-gpu', action='store_true', help='use GPU or not')


train_group.add_argument(
    '--optimizer',
    default='AdamW',
    choices=['SGD', 'Adam', 'AdamW'],
    help='optimizer choices')

#CHANGED-PARAM
train_group.add_argument(
    '--lr',
    type=float,
    default=0.005,
    metavar='F',
    help='initial learning rate')
train_group.add_argument(
    '--lr-decay',
    type=float,
    default=0.9,
    metavar='F',
    help='exponential decay of learning rate per lesson')
train_group.add_argument(
    '--accum-grad',
    type=int,
    default=1,
    metavar='N',
    help='accumulated gradient (default: 1)')
#CHANGED-PARAM
train_group.add_argument(
    '--batch-size',
    type=int,
    default=4,
    metavar='N',
    help='batch size for extra prediction')
train_group.add_argument(
    '--candidate-relax',
    type=int,
    default=0,
    metavar='N',
    help='number of thresh relaxation for candidate')

rl_group = parser.add_argument_group('Reinforcement Learning')

#CHANGED-PARAM
rl_group.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='F',
    help='discount factor for accumulated reward function in reinforcement learning'
)
rl_group.add_argument(
    '--penalty',
    type=float,
    default=-0.01,
    metavar='F',
    help='a small penalty each step')
rl_group.add_argument(
    '--entropy-beta',
    type=float,
    default=0.2,
    metavar='F',
    help='entropy loss scaling factor')

#CHANGED-PARAM
rl_group.add_argument(
    '--entropy-beta-decay',
    type=float,
    default=0.8,
    metavar='F',
    help='entropy beta exponential decay factor')

io_group = parser.add_argument_group('Input/Output')
io_group.add_argument(
    '--dump-dir', default=None, metavar='DIR', help='dump dir')
io_group.add_argument(
    '--dump-play',
    action='store_true',
    help='dump the trajectory of the plays for visualization')
io_group.add_argument(
    '--dump-fail-only', action='store_true', help='dump failure cases only')
io_group.add_argument(
    '--load-checkpoint',
    default=None,
    metavar='FILE',
    help='load parameters from checkpoint')

schedule_group = parser.add_argument_group('Schedule')
schedule_group.add_argument(
    '--runs', type=int, default=1, metavar='N', help='number of runs')
schedule_group.add_argument(
    '--early-drop-epochs',
    type=int,
    default=1000,
    metavar='N',
    help='epochs could spend for each lesson, early drop')
schedule_group.add_argument(
    '--save-interval',
    type=int,
    default=10,
    metavar='N',
    help='the interval(number of epochs) to save checkpoint')
schedule_group.add_argument(
    '--test-interval',
    type=int,
    default=None,
    metavar='N',
    help='the interval(number of epochs) to do test')
schedule_group.add_argument(
    '--test-only', action='store_true', help='test-only mode')
schedule_group.add_argument(
    '--test-not-graduated',
    action='store_true',
    help='test not graduated models also')

args = parser.parse_args()

args.use_gpu = args.use_gpu and torch.cuda.is_available()
args.concat_worlds = not args.no_concat_worlds
args.dump_play = args.dump_play and (args.dump_dir is not None)

if args.dump_dir is not None:
  io.mkdir(args.dump_dir)
  args.log_file = os.path.join(args.dump_dir, 'log.log')
  set_output_file(args.log_file)
else:
  args.checkpoints_dir = None
  args.summary_file = None

if args.seed is not None:
  random.reset_global_seed(args.seed)

make_env = functools.partial(
    make_env, random_order=True, exclude_self=True, fix_ground=True)

logger = get_logger(__file__)


def get_configuration(player, mode):
    if(mode=='initial'):
        bs = player.start_world.blocks
    else:
        assert(mode=='target')
        bs = player.final_world.blocks
    nblocks = len(bs)-1
    config = list(np.ones(nblocks,dtype=np.int64)*nblocks)
    for i in range(len(bs)):
        if(i>0):
            mrep = rev_perm(bs.random_order)
            father_i = mrep[bs._blocks[i].father.index]-1
            if(father_i>=0):
                config[father_i] = mrep[i]-1
            #print(mrep[i],'on top of', mrep[bs._blocks[i].father.index])
    return config


class Model(nn.Module):
  """The model for blocks world tasks."""

  def __init__(self):
    super().__init__() 

    # The 4 dimensions are: world_id, block_id, coord_x, coord_y
    input_dim = 4#+2
    self.transform = InputTransform('cmp', exclude_self=False)

    # current_dim = 4 * 3 = 12
    current_dim = transformed_dim = self.transform.get_output_dim(input_dim)
    self.feature_axis = 1 if args.concat_worlds else 2

    if args.model == 'memnet':
      self.feature = MemoryNet.from_args(
          current_dim, self.feature_axis, args, prefix='memnet')
      current_dim = self.feature.get_output_dim()
    else:
      input_dims = [0 for _ in range(args.nlm_breadth + 1)]
      input_dims[2] = current_dim
      self.features = LogicMachine.from_args(
          input_dims, args.nlm_attributes, args, prefix='nlm')
      current_dim = self.features.output_dims[self.feature_axis]

    self.final_transform = InputTransform('concat', exclude_self=False)
    if args.concat_worlds:
      current_dim = (self.final_transform.get_output_dim(current_dim) +
                     transformed_dim) * 2
    self.pred_valid     = LogicInference(current_dim, 1, [])
    self.pred_valid_2   = LogicInference(current_dim, 1, [])
    self.pred = LogitsInference(current_dim, 1, [])
    self.loss = REINFORCELoss()
    self.pred_loss = nn.BCELoss()
    self.num = None
    self.optim=None

  def forward(self, feed_dict, redundancy=False, index=None):
    feed_dict = GView(feed_dict)
    states = feed_dict.states.float()
    f = self.get_binary_relations(states)
    A = self.pred(f)
    logits = self.pred(f).squeeze(dim=-1).view(states.size(0), -1)
    policy = F.softmax(logits, dim=-1).clamp(min=1e-20)
    if not self.training:
      return dict(policy=policy, logits=logits)
    pred_states = feed_dict.pred_states.float()
    f = self.get_binary_relations(pred_states, depth=args.pred_depth)
    f = self.pred_valid(f).squeeze(dim=-1).view(pred_states.size(0), -1)
    # Set minimal value to avoid loss to be nan.
    valid = f[range(pred_states.size(0)), feed_dict.pred_actions].clamp(
        min=1e-20)
    pred_states_2 = feed_dict.pred_states_2.float()
    #f = self.get_binary_relations(pred_states_2, depth=self.pred_depth_2)
    f = self.get_binary_relations(pred_states_2, depth=args.pred_depth)
    f = self.pred_valid_2(f).squeeze(dim=-1).view(pred_states_2.size(0), -1)
    # Set minimal value to avoid loss to be nan.
    valid_2 = f[range(pred_states_2.size(0)), feed_dict.pred_actions_2].clamp(
        min=1e-20)
    
    loss, monitors = self.loss(policy, feed_dict.actions,
                               feed_dict.discount_rewards,
                               feed_dict.succ, feed_dict.number, feed_dict.entropy_beta, optim=self.optim, redundancy=redundancy, index=index)
    pred_loss = self.pred_loss(valid, feed_dict.valid)
    pred_loss_2 = self.pred_loss(valid_2, feed_dict.valid_2)
    
    monitors['pred/accuracy'] = feed_dict.valid_2.eq(
        (valid_2 > 0.5).float()).float().mean()
    return loss, monitors, dict()
    
  def get_binary_relations(self, states, depth=None):
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


def make_data(traj, gamma, succ, number, normalization=1):
  """Aggregate data as a batch for RL optimization."""
  q = 0
  discount_rewards = []
  for reward in traj['rewards'][::-1]:
    q = q * gamma + reward
    discount_rewards.append(q)
  discount_rewards.reverse()
  for i in range(len(discount_rewards)):
      discount_rewards[i] /= normalization
      #discount_rewards[i] *= 2
      if(not succ):
          ...
          discount_rewards[i] /= 20
      else: 
          ...
  traj['states'] = as_tensor(np.array(traj['states']))
  traj['actions'] = as_tensor(np.array(traj['actions']))
  traj['discount_rewards'] = as_tensor(np.array(discount_rewards)).float()
  traj['succ'] = succ
  traj['number'] = number
  return traj



reward_data = {i:[] for i in range(1,13)}

def run_episode_cpbp(env,
                actions_history,
                number,
                model_baseline=None):
  need_restart=False
  """Run one episode using the model with $number blocks."""
  is_over = False
  traj = collections.defaultdict(list)
  score = 0
  if need_restart:
    env.restart()
  env.unwrapped.run_mode = 'train'#run_mode 
  nr_objects = number + 1
  rs = []
  env.unwrapped.model = model_baseline#model
  env.unwrapped.mapping = env.mapping
  env.unwrapped.initiate_distribution()
  estimates_history=[]
  zzzz = 0
  #print(actions_history)
  for action in actions_history:
    zzzz += 1
    
    mapped_x, mapped_y = env.mapping[action]
    reward, is_over = env.action(action)
    score += reward
    rs += [reward]
    _, succ = env.unwrapped._get_result()
    traj['rewards'].append(reward)
    if(is_over):
        ...
        break
  estimates_history = env.unwrapped.rf.es
  estimates_history = env.unwrapped.es
  es_distributions = env.unwrapped.es_ds
  length = len(traj['rewards'])
  
  def get_return(rews):
    q = 0
    discount_rewards = []
    for reward in rews[::-1]:
      q = q + reward
      discount_rewards.append(q)
    discount_rewards.reverse()
    return discount_rewards
  return succ, traj, score, env.unwrapped.baseline_lengths, estimates_history, None, es_distributions#get_return(env.unwrapped.rs)

def get_expected_value(distribution, threshold=1.0, flat_mode=False, n_actions_taken=0):
    if(flat_mode):
        r = 1.0/len(list(distribution.keys()))
        for v in distribution.keys():
            distribution[v] = r
    
    e=0
    cumul = 0
    #print(distribution.keys())
    vals = np.sort(list(distribution.keys()))
    #print(vals)
    for v in vals: 
        if(cumul + distribution[v] < threshold):
            e += distribution[v]*(v-n_actions_taken)
            cumul = cumul + distribution[v]
        else:
            e+=(threshold - cumul)*(v-n_actions_taken);
            break
    return e/threshold
    



def run_episode(env,
                model,
                number,
                play_name='',
                dump=False,
                dataset=None,
                dataset_2=None,
                eval_only=False,
                use_argmax=False,
                need_restart=False,
                entropy_beta=0.0,
                run_mode='train',
                model_baseline=None):
  """Run one episode using the model with $number blocks."""
  is_over = False
  traj = collections.defaultdict(list)
  score = 0
  if need_restart:
    env.restart()
  env.unwrapped.run_mode = 'run'#run_mode 
  #print('baselines:',env.unwrapped.baseline_lengths.keys())
  model.num = env.unwrapped.nr_blocks
  nr_objects = number + 1
  # If dump_play=True, store the states and actions in a json file
  # for visualization.
  dump_play = args.dump_play and dump
  if dump_play:
    array = env.unwrapped.current_state
    moves, new_pos, policies = [], [], []
  rs = []
  env.unwrapped.model = model_baseline#model
  env.unwrapped.mapping = env.mapping
  env.unwrapped.initiate_distribution()
  r_env = env.unwrapped.get_string_representation()
  states_history = []
  while not is_over:
    state = env.current_state
    
    feed_dict = dict(states=np.array([state]))
    feed_dict['entropy_beta'] = as_tensor(entropy_beta).float()
    feed_dict = as_tensor(feed_dict)
    if args.use_gpu:
      feed_dict = as_cuda(feed_dict)
    with torch.set_grad_enabled(not eval_only):
      output_dict = model(feed_dict)
      
    policy = output_dict['policy']
    p = as_numpy(policy.data[0])
    rrr = random.rand()
    #if(rrr>0.25):
    #    use_argmax = True
    n_actions = len(p)
    action = p.argmax() if use_argmax else random.choice(len(p), p=p)



      
    '''
    action_distribution = []
    for i in range(len(p)):
        action_distribution += [(env.mapping[i], p[i])]
    '''
    #print(action_distribution)
    # Need to ensure that the env.utils.MapActionProxy is the outermost class.
    mapped_x, mapped_y = env.mapping[action]
    # env.unwrapped to get the innermost Env class.
    valid = env.unwrapped.world.moveable(mapped_x, mapped_y)
    valid_2 = env.unwrapped.world.moveable_2(mapped_x, mapped_y)
    #env.unwrapped.action_distribution = action_distribution
    
    reward, is_over = env.action(action)
    #if(len(estimates_history)==0):
    #    estimates_history += [env.unwrapped.rf.prev_exp_cost]
    rs += [reward]
    if dump_play:
      moves.append([mapped_x, mapped_y])
      res = tuple(env.current_state[mapped_x][2:])
      new_pos.append((int(res[0]), int(res[1])))

      logits = as_numpy(output_dict['logits'].data[0])
      tops = np.argsort(p)[-10:][::-1]
      tops = list(
          map(lambda x: (env.mapping[x], float(p[x]), float(logits[x])), tops))
      policies.append(tops)

    # For now, assume reward=1 only when succeed, otherwise reward=0.
    # Manipulate the reward and get success information according to reward.
    #reward += args.penalty
    '''
    if reward == 0 and args.penalty is not None:
      reward = args.penalty
    succ = 1 if is_over and reward > 0.99 else 0
    '''
    _, succ = env.unwrapped._get_result()
    
    score += reward
    #print('reward:',reward)
    traj['states'].append(state)
    traj['rewards'].append(reward)
    traj['actions'].append(action)
    if not eval_only and dataset is not None and mapped_x != mapped_y:
      dataset.append(nr_objects, state, action, valid)
      dataset_2.append(nr_objects, state, action, valid_2)
  #print('rewards:',rs)
  rs_list = rs
  rs = np.array(rs)
  if((not succ) and (score>0)):
        ...
        #print('Anomaly:',rs_list)
        
  for i in range(1,len(rs)):
    rs[len(rs)-1-i] += rs[len(rs)-i]
  if(run_mode=='train'):
    reward_data[number] += [(rs, succ, )]  
#print('returns:',rs)
  # Dump json file as record of the playing.
  if dump_play and not (args.dump_fail_only and succ):
    array = array[:, 2:].astype('int32').tolist()
    array = [array[:nr_objects], array[nr_objects:]]
    json_str = json.dumps(
        # Let indent=True for an indented view of json files.
        dict(array=array, moves=moves, new_pos=new_pos,
             policies=policies))
    dump_file = os.path.join(
        args.current_dump_dir,
        '{}_blocks{}.json'.format(play_name, env.unwrapped.nr_blocks))
    with open(dump_file, 'w') as f:
      f.write(json_str)
  
  length = len(traj['rewards'])
  return succ, score, traj, length, n_actions


class MyTrainer(MiningTrainerBase):
  def initiate_DQN(self):
    self.DQN = DQN(args)
  def save_checkpoint(self, name):
    if args.checkpoints_dir is not None:
      checkpoint_file = os.path.join(args.checkpoints_dir,
                                     'checkpoint_{}.pth'.format(name))
      super().save_checkpoint(checkpoint_file)
      checkpoint_file = os.path.join(args.checkpoints_dir,
                                     'dataset_{}.pth'.format(name))  
      #with open(checkpoint_file, 'wb') as handle:
        #pickle.dump((self.pos_data, self.neg_data, self.lr, self.current_epoch, self.entropy_beta, self.current_number, self.sample_array, self.valid_action_dataset, self.valid_action_dataset_2), handle, protocol=pickle.HIGHEST_PROTOCOL)
            #(self.pos_data, self.neg_data, trainer.lr, trainer.current_epoch, trainer.entropy_beta)
  def _dump_meters(self, meters, mode):
    if args.summary_file is not None:
      meters_kv = meters._canonize_values('avg')
      meters_kv['mode'] = mode
      meters_kv['epoch'] = self.current_epoch
      meters_kv['length_std'] = meters['length'].std
      meters_kv['succ_std']   = meters['succ'].std
      meters_kv['loss_std']   = meters['loss'].std
      meters_kv['number_std']   = meters['number'].std
      meters_kv['reinforce_loss_std']   = meters['reinforce_loss'].std
      meters_kv['entropy_std']   = meters['entropy'].std
      meters_kv['entropy_loss_std']   = meters['entropy_loss'].std
      meters_kv['pred/accuracy_std']   = meters['pred/accuracy'].std
      with open(args.summary_file, 'a') as f:
        f.write(io.dumps_json(meters_kv))
        f.write('\n')

  def _prepare_dataset(self, epoch_size, mode):
    pass

  def _get_player(self, number, mode):
    player = make_env(args.task, number)
    player.restart()
    return player

  def _get_result_given_player(self, index, meters, number, player, mode):
    assert mode in ['train', 'test', 'mining', 'inherit']
    params = dict(
        eval_only=True,
        number=number,
        play_name='{}_epoch{}_episode{}'.format(mode, self.current_epoch,
                                                index),
        run_mode=mode,
        model_baseline=self.model)#self.trainer_baseline.model)
    backup = None
    if mode == 'train':
      params['eval_only'] = False
      params['dataset'] = self.valid_action_dataset
      params['dataset_2'] = self.valid_action_dataset_2
      params['entropy_beta'] = self.entropy_beta
      meters.update(lr=self.lr, entropy_beta=self.entropy_beta)
    elif mode == 'test':
      params['dump'] = True 
      params['use_argmax'] = True
    else:
      backup = copy.deepcopy(player)
      params['use_argmax'] = self.is_candidate
    LENGTH = 4*player.unwrapped.nr_blocks 
    player.proxy.set_limit(LENGTH) #Episode finishes within LENGTH steps, successful or not.
    player_copy = copy.deepcopy(player)
    succ, score, traj, length, n_actions= run_episode(player, self.model, **params)
    baseline_len = len(traj['actions'])
    player_copy.unwrapped.max_len=2*LENGTH #Sets the number of steps of the CP model used to compute the state potentials (this is a hyperparameter). 
    if(succ):
        player_copy.unwrapped.succeeded = True
    else:
        player_copy.unwrapped.succeeded = False
    player_copy.unwrapped.limit_len = LENGTH
    def get_rets(traj):
        q = 0
        rets = []
        for reward in traj['rewards'][::-1]:
            q = q + reward
            rets.append(q)
        rets.reverse()
        return rets
    
    vanilla_mode=False
    use_cpbp_fail = True
    
    if(mode=='train' and vanilla_mode or ((not succ) and (not use_cpbp_fail))):
        for i in range(len(traj['rewards'])-1):
            traj['rewards'][i] = args.penalty
        if(not succ):
            traj['rewards'][-1]= args.penalty
        if(succ):
            traj['rewards'][-1] = 1
        score = np.sum(traj['rewards'])
    baseline_lengths= None
    USED_LENGTH = len(traj['states'])
    if((not vanilla_mode) and mode=='train' and (succ or use_cpbp_fail)):
        if(not succ):
            ...
            #player_copy.unwrapped.threshold = 1.0
        succ_cpbp, traj_cpbp, score_cpbp, baseline_lengths, estimates_history, true_es, es_distributions = run_episode_cpbp(player_copy, traj['actions'], number, model_baseline=self.model)
        traj['rewards']               = traj_cpbp['rewards']
        traj['es_distributions']      = es_distributions
        traj['distribution_threshold']= player_copy.unwrapped.threshold
        
        #print(traj['es_distributions'][0])
        
        
        for i in range(len(estimates_history)):
            estimates_history[i] += i 
        if(not succ):
            traj['states']  = traj['states'][:len(traj['rewards'])]
            traj['rewards'] = traj['rewards'][:len(traj['rewards'])]
            traj['actions'] = traj['actions'][:len(traj['rewards'])]
    
    if mode == 'train':
      norm=None
      norm=number
      feed_dict = make_data(traj, args.gamma, succ, number, normalization=norm)
      traj=feed_dict
      score = np.mean(np.array(traj['discount_rewards'], dtype=np.float64))
      feed_dict['entropy_beta'] = as_tensor(self.entropy_beta).float()

      # content from valid_move dataset
      states, actions, labels = \
          self.valid_action_dataset.sample_batch(args.batch_size)
      states_2, actions_2, labels_2 = \
          self.valid_action_dataset_2.sample_batch(args.batch_size)
      feed_dict['pred_states'] = as_tensor(states)
      feed_dict['pred_actions'] = as_tensor(actions)
      feed_dict['valid'] = as_tensor(labels).float()
      feed_dict['pred_states_2'] = as_tensor(states_2)
      feed_dict['pred_actions_2'] = as_tensor(actions_2)
      feed_dict['valid_2'] = as_tensor(labels_2).float()
      feed_dict['min_threshold'] = self.min_threshold
        
      if args.use_gpu:
        feed_dict = as_cuda(feed_dict)
      meters.update(number=number, succ=succ, score=score, length=length)
      ess = [(get_expected_value(traj['es_distributions'][i][2], threshold=1.0, n_actions_taken=i, flat_mode=True)) for i in range(len(traj['es_distributions'])-1)]
      return feed_dict, ess
    else:
      meters.update(number=number, succ=succ, score=score, length=length)
      message = ('> {} iter={iter}, number={number}, succ={succ}, '
                 'score={score:.4f}, length={length}').format(
                     mode, iter=index, **meters.val)
      return message, dict(succ=succ, number=number, backup=backup)

  def _extract_info(self, extra):
    return extra['succ'], extra['number'], extra['backup']

  def _get_accuracy(self, meters):
    return meters.avg['succ']

  def _get_threshold(self):
    candidate_relax = 0 if self.is_candidate else args.candidate_relax
    #print('candidate_relax:',args.candidate_relax)
    return super()._get_threshold() - \
        self.curriculum_thresh_relax * candidate_relax

  def _upgrade_lesson(self):
    super()._upgrade_lesson()
    # Adjust lr & entropy_beta w.r.t different lesson progressively.
    self.lr *= args.lr_decay
    self.entropy_beta *= args.entropy_beta_decay
    self.set_learning_rate(self.lr)

  def _train_epoch(self, epoch_size):
    meters = super()._train_epoch(epoch_size)
    i = self.current_epoch
    if args.save_interval is not None and i % args.save_interval == 0:
      self.save_checkpoint(str(i))
    if args.test_interval is not None and i % args.test_interval == 0:
      self.test()

    return meters

  def _early_stop(self, meters):
    t = args.early_drop_epochs
    if t is not None and self.current_epoch > t * (self.nr_upgrades + 1):
      return True
    return super()._early_stop(meters)

  def train(self):
    #super().save_checkpoint(os.path.join(args.checkpoints_dir,'best_trainer.pth'))
    #if(self.valid_action_dataset==None):
    self.valid_action_dataset = ValidActionDataset()
    self.valid_action_dataset_2 = ValidActionDataset()
    #else:
    #    print('loaded preds dataset...')
    if(self.lr is None):
        self.lr = args.lr
    if(self.entropy_beta is None):
        self.entropy_beta = args.entropy_beta
    return super().train()


def main(run_id,load_first):
  if args.dump_dir is not None:
    if args.runs > 1:
      args.current_dump_dir = os.path.join(args.dump_dir,
                                           'run_{}'.format(run_id))
      io.mkdir(args.current_dump_dir)
    else:
      args.current_dump_dir = args.dump_dir
    args.checkpoints_dir = os.path.join(args.current_dump_dir, 'saves')
    io.mkdir(args.checkpoints_dir)
    args.summary_file = os.path.join(args.current_dump_dir, 'summary.json')

  logger.info(format_args(args))

  model = Model()
  if args.use_gpu:
    model.cuda()
  optimizer = get_optimizer(args.optimizer, model, args.lr)
  if args.accum_grad > 1:
    optimizer = AccumGrad(optimizer, args.accum_grad)
  model_2 = Model()
  if args.use_gpu:
    model_2.cuda()
  optimizer_2 = get_optimizer(args.optimizer, model_2, args.lr)
  if args.accum_grad > 1:
    optimizer_2 = AccumGrad(optimizer_2, args.accum_grad)
  #print('A')
  #print('has attrib:',hasattr(model, 'train_step'))
  trainer = MyTrainer.from_args(model, optimizer, args)
  trainer.optim = optimizer
  trainer_baseline = MyTrainer.from_args(model_2, optimizer_2, args)
  trainer.trainer_baseline = trainer_baseline
  trainer.initiate_DQN()
  trainer.checkpoints_dir = args.checkpoints_dir
  model.optim = optimizer
  trainer_baseline.checkpoints_dir = args.checkpoints_dir
  if args.load_checkpoint is not None:
    if(load_first):
        print("Loading!")
        trainer.load_checkpoint(args.load_checkpoint)
        cc = '/'.join(args.load_checkpoint.split('/')[:-1])
        
        #checkpoint_file = os.path.join(cc, 'dataset_{}.pth'.format(str(200)))  
        #with open(checkpoint_file, 'rb') as handle:
        #    (trainer.pos_data, trainer.neg_data, trainer.lr, trainer.current_epoch, trainer.entropy_beta, trainer.current_number,  trainer.sample_array, trainer.valid_action_dataset, trainer.valid_action_dataset_2) = pickle.load(handle)
            #(_, _, trainer.lr, trainer.current_epoch, trainer.entropy_beta, trainer.current_number,  trainer.sample_array, trainer.valid_action_dataset, trainer.valid_action_dataset_2) = pickle.load(handle)
            #(_, _, trainer.lr, trainer.current_epoch, trainer.entropy_beta, trainer.current_number, _) = pickle.load(handle)
        #trainer.current_number += 1
        #trainer.sample_array = [7]

        #print('loaded entropy_beta:', trainer.entropy_beta)   
        #trainer.epochs = 1000
        #trainer.entropy_beta = 0.08192*0.5
        #trainer.entropy_beta *= 0.5
        #trainer.lr = 0.0001
        #(trainer.pos_data, trainer.neg_data, trainer.lr, trainer.current_epoch, trainer.entropy_beta, trainer.current_number, trainer.sample_array) = pickle.load(handle)
        #trainer.nr_upgrades += trainer.current_number-trainer.
        #print('num:',trainer.current_number)
        #trainer.set_learning_rate(trainer.lr)
        #trainer.last_mining_epoch = trainer.current_epoch
        #trainer.current_epoch += 1
        
    else:
        print("Did not load!")


  if args.test_only:
    trainer.current_epoch = 0
    return None, trainer.test()

  graduated = trainer.train()
  trainer.save_checkpoint('last')
  #test_meters = trainer.test() if graduated or args.test_not_graduated else None
  return graduated, None#test_meters


if __name__ == '__main__':
  stats = []
  nr_graduated = 0
  #1087, 1091,1093,1097,1117,1123,1129,1151
  seed_list  = [1087,1091,1093,1097,1117,11231129,1151,1163,1153]
  #seed_list  = [1117,1129,1151,1163,1153]
  #seed_list  =  [1163,1153]
  load_first = True
  for s in seed_list:
    
    #args.dump_dir = 'training_cpbp_expensive_flat_domainchange_'+str(s)
    args.dump_dir = 'training_cpbp_'+str(s)
    io.mkdir(args.dump_dir)
    args.log_file = os.path.join(args.dump_dir, 'log.log')
    set_output_file(args.log_file)
    random.reset_global_seed(s)
    i=0
    #graduated, test_meters = main(i)
    #graduated, test_meters = main(i,load_first)
    load_first = False
    graduated, test_meters = main(i,load_first)
    '''
    try:
        graduated, test_meters = main(i,load_first)
        load_first = False
    except:
        print("FAILURE")
    '''
