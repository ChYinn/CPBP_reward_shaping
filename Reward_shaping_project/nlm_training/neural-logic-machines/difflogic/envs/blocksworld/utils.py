from difflogic.envs.blocksworld import envs
from difflogic.envs.blocksworld.represent import get_coordinates
from difflogic.envs.utils import get_action_mapping_blocksworld
from automatons import  action_to_index_global
import requests
import numpy as np
import math
from collections import deque 
import pickle
import os
class Reward_Fetcher():
    def __init__(self, player, length_multiplier=-1, gamma=.99, plan_size_try=0, free_space=0, max_stacks=0,  target_stacks = 0, max_len=0, threshold=-1, action_distribution=None,
                baseline_lengths=None):
        #print('made RWF')
        self.baseline_lengths = baseline_lengths
        self.action_distribution = action_distribution
        self.player = player
        self.max_len = max_len
        self.threshold = threshold
        self.length_multiplier = length_multiplier
        self.c_init       = get_configuration(player, 'initial')
        self.c_target     = get_configuration(player, 'target')
        self.c_init_str   = get_configs_str(self.c_init)
        self.c_target_str = get_configs_str(self.c_target)
        self.nb_blocks    = len(self.c_init)
        self.plan_size_try = plan_size_try
        self.target_stacks = target_stacks
        self.actions      = []
        self.nErrors      = 0
        self.exp_cost     = 0
        self.prev_exp_cost= 0
        self.min_cost     = 0
        self.entropy      = 0
        self.free_space   = free_space
        self.max_stacks   = max_stacks
        r, is_over        = player._get_result()
        self.is_over      = is_over
        self.offset       = player.nActions
        self.es_distribution = {} 
        #self.exp_cost, self.min_cost, self.entropy, self.inconsistent, es_distribution_expensive = self.compute_costs(mode='expensive')
        self.mode = 'expensive'
        self.exp_cost, self.min_cost, self.entropy, self.inconsistent, es_distribution_cheapest = self.compute_costs(mode=self.mode)
        #print(self.c_init_str,'to',self.c_target_str,es_distribution_cheapest)
        #e_cheapest, _, _, _, es_distribution_cheapest = self.compute_costs(mode='cheapest')
        #_, _, _, _, es_distribution_mid       = self.compute_costs(mode='mid')
        #_, _, _, _, es_distribution_expensive = self.compute_costs(mode='expensive')
        #self.exp_cost = e_cheapest
        es_distribution_mid, es_distribution_expensive = {0:0}, {0:0}
        self.es_distribution = (es_distribution_expensive,es_distribution_mid,es_distribution_cheapest)
        '''
        distribution_data = {n:[] for n in np.arange(2,13)}
        if(os.path.isfile("distribution_data.pickle")):
            with open(r"distribution_data.pickle", "rb") as input_file:
                distribution_data = pickle.load(input_file)
        #print(distribution_data)
        distribution_data[self.nb_blocks] += [self.es_distribution]
        with open(r"distribution_data.pickle", "wb") as output_file:
            pickle.dump(distribution_data, output_file)
        '''
        #print(self.es_distribution)
        #self.es_distribution = ({0:0},{0:0},es_distribution_cheapest)
        #print([e_cheapest,e_mid,self.exp_cost])
        #self.append_approximation_cost(self.exp_cost)
        #self.exp_cost = self.get_approximation_cost()
        self.base_exp = self.exp_cost
        
        #self.exp_cost, self.min_cost = self.compute_costs()
        self.gamma = gamma
        self.rs = []
        self.last_potential = self.get_potential()
        self.es = [-(self.exp_cost)]#-len(self.actions)-self.length_multiplier)]
    def get_potential(self):
        return (-1*self.entropy+1)*(self.exp_cost)
    
    def compute_costs(self, mode='cheapest'):
        #return 0,0,0,0
        actions_str ='#'.join([str(j) for j in self.actions])
        #print(actions_str
        self.player.action_distribution=[]
        action_distribution_str = '#'.join([str((self.action_translate(j[0]),j[1])) for j in self.player.action_distribution])
        #self.offset = len(self.actions)
        #print('plan_size_try:',str(self.plan_size_try))
        msg = requests.post('http://172.30.128.1:5000/solve_configuration', json={"nBlocks":str(self.nb_blocks), 
                                                                                  "config_initial":self.c_init_str, 
                                                                                  "config_target":self.c_target_str,
                                                                                  'actions':actions_str,
                                                                                  'action_distribution':action_distribution_str,
                                                                                  'isOver':str(self.is_over),
                                                                                  'length_multiplier':str(self.offset),
                                                                                  'plan_size_try':str(self.plan_size_try),
                                                                                  'free_space':str(self.free_space),
                                                                                  'max_stacks':str(self.max_stacks),
                                                                                  'target_stacks':str(self.target_stacks),
                                                                                  'max_len':str(self.max_len),
                                                                                  'threshold':str(self.threshold),
                                                                                  'mode':mode}, timeout=None).text
        exp_cost, min_cost, entropy, inconsistent, distribution_str = None, None, None, None, None
        exp_cost, min_cost, entropy, inconsistent, distribution_str = [a for a in msg.split(',')]
        exp_cost, min_cost, entropy, inconsistent = float(exp_cost), float(min_cost), float(entropy), float(inconsistent)

        distribution_str = [v.split(":") for v in distribution_str.split("#")]
        distribution = {}
        for i in range(len(distribution_str)):
            #print(distribution_str[i])
            if(not (distribution_str[i]==[''] or distribution_str[i]==['nan'])):
                distribution[int(distribution_str[i][0])] = float(distribution_str[i][1])
        
        def get_expected_value(distribution):
            e=0
            for v in distribution.keys():
                e += distribution[v]*v
            #print(e)
            return e
        #print(get_expected_value(distribution), exp_cost)
        #exp_cost += self.offset
        #min_cost += self.offset
        return exp_cost, min_cost, entropy, inconsistent, distribution
    def append_approximation_cost(self,exp_cost):
        state_str = self.player.get_string_representation()
        if(not state_str in self.baseline_lengths[self.nb_blocks].keys()):
            #self.baseline_lengths[self.nb_blocks][state_str] = deque(maxlen=10)
            self.baseline_lengths[self.nb_blocks][state_str] = exp_cost-len(self.actions)#deque(maxlen=20)
            
        #self.baseline_lengths[self.nb_blocks][state_str].append(exp_cost-len(self.actions))
        self.baseline_lengths[self.nb_blocks][state_str] = self.baseline_lengths[self.nb_blocks][state_str]*0.25 + (exp_cost-len(self.actions))*0.75
        
    def get_approximation_cost(self):
        if(len(self.actions)==self.plan_size_try):
            return len(self.actions)
        state_str = self.player.get_string_representation()
        #exp_cost = np.mean(self.baseline_lengths[self.nb_blocks][state_str])+len(self.actions)
        exp_cost = self.baseline_lengths[self.nb_blocks][state_str] +len(self.actions)
        return exp_cost
    
    def move(self,a,b,ground_change):
        self.actions+=[(a,b,ground_change)]
        #self.actions+=[(a,b)]
        #exp_cost_cheapest, min_cost_cheapest, entropy_cheapest, inconsistent_cheapest,      es_distribution_cheapest = self.compute_costs(mode='cheapest')
        #exp_cost_expensive, min_cost_expensive, entropy_expensive, inconsistent_expensive, es_distribution_expensive = self.compute_costs(mode='expensive')
        exp_cost, min_cost, entropy, inconsistent,    es_distribution = self.compute_costs(mode=self.mode)
        self.es += [-(exp_cost)]#-len(self.actions)-self.length_multiplier)]
        
        if(inconsistent==1): 
            self.inconsistent = 1
        #self.append_approximation_cost(exp_cost)
        #print('maxlen',self.max_len,':',self.actions,'->',exp_cost)
        #exp_cost = self.get_approximation_cost()
        reward_exp_cost    = exp_cost - self.exp_cost 
        reward_min_cost    = min_cost - self.min_cost
        reward_entropy     = entropy  - self.entropy
        self.prev_exp_cost = self.exp_cost
        self.prev_min_cost = self.min_cost
        
        self.exp_cost = exp_cost
        self.min_cost = min_cost
        self.entropy  = entropy
        return reward_exp_cost, 0, #(es_distribution, es_distribution)
    
    def action_translate(self,ab_tuple):
        a = ab_tuple[0]
        b = ab_tuple[1]
        c=self.player.start_world.ground_change(a,b)
        a_ = None
        b_ = None
        if(b==0):
            b_=self.nb_blocks
        else:
            b_=b-1
        if(a==0):
            a_=self.nb_blocks
        else:
            a_=a-1
        x,y = None,None
        if(a_==self.nb_blocks):
            x = 0
            y = 0
        else:
            x=a_
            y=b_
        return (x,y,c)
    def move_raw(self,a,b,is_over,movable,ground_change):
        _, succ = self.player._get_result()
        is_over=succ
        self.is_over = is_over
        a_ = None
        b_ = None
        if(not movable):
            b_=0
            a_=0
        else:
            if(b==0):
                b_=self.nb_blocks
            else:
                b_=b-1
            if(a==0):
                a_=self.nb_blocks
            else:
                a_=a-1
        x,y = None,None
        if(a_==self.nb_blocks):
            x = 0
            y = 0
        else:
            x=a_
            y=b_
        return self.move(x,y,ground_change)

    
    
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
def get_configs_str(c):
    nBlocks=len(c)
    c = [str(i)+':'+str(c[i]) for i in range(len(c))]
    c_str = ''
    for i in range(len(c)):
        c_str+=c[i]
        if(i<len(c)-1):
            c_str+=','
    return c_str
def rev_perm(perm):
    mrep = list(np.zeros(len(perm)))
    for i in range(len(perm)):
        mrep[perm[i]] = i
    #mrep = list(np.arange(len(perm)))
    return mrep
def msg_to_actions(msg):
    #print(msg)
    msg = msg.split('#')
    action_msg = msg[0]
    stat_msg = msg[1]
    action_msg = action_msg.split(',')
    action_msg = [(int(m.split(':')[0]), int(m.split(':')[1])) for m in action_msg]
    
    stat_msg = stat_msg.split(',')
    s ={}
    for d in stat_msg: 
        s[d.split(':')[0]] = int(d.split(':')[1])
    
    return action_msg, s