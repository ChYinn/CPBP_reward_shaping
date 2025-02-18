import torch
from torch import nn
from torchrl.data.tensor_specs import Bounded
from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
from torchrl.modules.tensordict_module.common import SafeModule
from torchrl.objectives.ppo import PPOLoss
from tensordict import TensorDict
from dqn.nlm_pred import Model_NLM
import pickle

with open(r"arguments.pickle", "rb") as input_file:
    args= pickle.load(input_file)


nlm = Model_NLM(args)
net = nn.Sequential(nlm, NormalParamExtractor())
