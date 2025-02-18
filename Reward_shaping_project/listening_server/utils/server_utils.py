from automatons_constrained import BlocksWorld as BlocksWorld_expensive
from automatons_cheapest import BlocksWorld as  BlocksWorld_cheapest

import numpy as np
import requests
import math
def extract_config(config_str, nBlocks):
    config_str = config_str.split(',')
    config = np.zeros(nBlocks, dtype=np.int64)
    for i in range(nBlocks):
        c_i = config_str[i].split(':')
        #print(c_i)
        config[int(c_i[0])] = int(c_i[1])
    return config




def create_file(nBlocks, init, target, actions, isOver, plan_size_try, mode='train', length_multiplier=-1, free_space=1, max_stacks=0, target_stacks=0, max_len=0,thresh=1, distribution=None, model_mode='cheapest'):
    MAX_SIZE        = max_len#2*phi(4,nBlocks)#4*nBlocks
    N_ACTIONS_TAKEN = int(length_multiplier)
    #USE IF SOLVE_SIZE
    #planSize = int(plan_size_try)
    #USE IF 2X SIZE 
    planSize = min(plan_size_try, MAX_SIZE-N_ACTIONS_TAKEN)
    if(model_mode=='cheapest'):
        print('using '+model_mode)
        bw = BlocksWorld_cheapest(nBlocks,planSize,planSize)
        bw.init_config = init
        bw.goal_config = target
        bw.create_automatons()
    else:
        bw = BlocksWorld_expensive(nBlocks,planSize,planSize)
        bw.init_config = init
        bw.goal_config = target
        bw.create_automatons()
        if(model_mode=='expensive'):
            print('using '+model_mode)
            bw.create_counter_automaton(free_space, max_stacks,target_stacks)
        else:
            print('using '+model_mode)

    #print(bw.global_actions_keys)
    actions_ = []
    for i in range(len(actions)):
        if((actions[i][0], actions[i][1])==(0,0)):
            actions_ += [-1]#[bw.global_actions_keys['stay']]
        else:
            a,b,c = actions[i]
            if(a==nBlocks):
                assert(False)
                a = 'air'
            if(b==nBlocks):
                b = 'ground'
            if(c==0):
                c = 'no_change'
            elif(c==1):
                c = 'lifted'
            elif(c==2):
                c = 'added'
            if(b=='ground' and c=='no_change'):
                actions_ += [-1]
            else:
                if(model_mode=='cheapest'):
                    actions_ += [bw.global_actions_keys[(a,b,'dummy')]]
                else:
                    actions_ += [bw.global_actions_keys[(a,b,c)]]
    #print('actions_:',actions_)
    
    if(isOver):
        for i in range(len(actions),planSize):
            actions_ += [bw.global_actions_keys['finish']]
    
    actions = actions_
    directory = 'MiniCPBP/src/main/java/minicpbp/examples/data/ClassicalAIplanning/'
    filename  = 'problem.txt'
    with open(directory + filename, "w") as f:
        f.write(bw.render())
    if(mode=='train'):
        parameters_str = ''
        parameters_str += str(planSize) + '\n'
        parameters_str += str(nBlocks)  + '\n'
        parameters_str += str(len(actions))  + '\n'
        for i in range(len(actions)):
            parameters_str += str(actions[i]) 
            if(i<len(actions)-1):
                parameters_str += ' '
        parameters_str += '\n'
        for i in range(len(init)):
            parameters_str += str(init[i]) 
            if(i<len(init)-1):
                parameters_str += ' '
        parameters_str += '\n'
        for i in range(len(target)):
            parameters_str += str(target[i]) 
            if(i<len(target)-1):
                parameters_str += ' '
        parameters_str += '\n' + str(thresh)
        distribution_raw = np.zeros(len(bw.global_actions))
        action_invalid_prob = 0
        for i in range(len(distribution)):
            action_a = (int(distribution[i][0]), int(distribution[i][1]), int(distribution[i][2]))
            a,b,c = action_a
            if(a==nBlocks):
                assert(False)
                a = 'air'
            if(b==nBlocks):
                b = 'ground' 
            if(c==0):
                c = 'no_change'
            elif(c==1):
                c = 'lifted'
            elif(c==2):
                c = 'added'
            action_a = (a,b,c)
            if(not (a,b) == (0,0)):
                distribution_raw[bw.global_actions_keys[action_a]] += distribution[i][3]
            else:
                action_invalid_prob += distribution[i][2]
        #print(distribution_raw)
        parameters_str += '\n' + ' '.join([str(d) for d in distribution_raw])
        filename  = 'parameters.txt'
        with open(directory + filename, "w") as f:
            f.write(parameters_str)
    else:
        assert(mode=='solve','invalid mode!')
    return actions, planSize
def solve_file():
    x = requests.get('http://localhost:9005/solve', timeout=None)
    return x.text
def read_statistics():
    directory = 'MiniCPBP/src/main/java/minicpbp/examples/data/ClassicalAIplanning/'
    filename  = 'statistics.txt'
    data=''
    with open(directory+filename) as f:
        data = f.read()
    return data
    data = data.split(',')
    s ={}
    for d in data: 
        s[d.split(':')[0]] = d.split(':')[1]
    return s
def read_rewards(init, target, actions, N_ACTIONS_TAKEN):
    directory = 'MiniCPBP/src/main/java/minicpbp/examples/data/ClassicalAIplanning/'
    filename  = 'rewards.txt'
    data=''
    with open(directory+filename) as f:
        data = f.read()
    data = data.split('\n')
    init_read = [int(b) for b in data[0].split(',')]
    target_read = [int(b) for b in data[1].split(',')]
    nActions_read = int(data[2])
    actions_read  = []
    if(nActions_read>0):
        actions_read = [int(b) for b in data[3].split(',')]
    #print(init, target)
    #print(init_read, target_read, actions_read)
    try: 
        assert len(init) == len(init_read)
        assert len(target) == len(target_read)
        #assert len(actions) == len(actions_read)
        for i in range(len(init)):
            assert(init_read[i]==init[i])
        for i in range(len(target)):
            assert(target_read[i]==target[i])
        #for i in range(len(actions)):
        #    assert(actions_read[i]==actions[i])
    except:
        print('Read configuration mismatch, retrying...')
        print('init:',init,'read:',init_read)
        print('target:',target,'read:',target_read)
        print('actions:',actions,'read:',actions_read)
        return None, None
    expected_cost, minimum_cost, entropy = data[3+(len(actions_read)>0)].split(',')
    distribution_data = read_distribution(N_ACTIONS_TAKEN)
    return float(expected_cost), int(minimum_cost), float(entropy), distribution_data
    
def read_solution(nBlocks):
    directory = 'MiniCPBP/src/main/java/minicpbp/examples/data/ClassicalAIplanning/'
    filename  = 'solution.txt'
    data=''
    with open(directory+filename) as f:
        data = f.read()
    data = [int(d) for d in data.split(',')]
    
    def index_to_action_global(index, nbBlocks):
        """
        returns the move(i,j) of the index 
        """
        if(index>=0 and index < nbBlocks*(nbBlocks)):
            i_ = int(index/(nbBlocks))
            j_ = index%(nbBlocks)
            index_ = i_*(nbBlocks)+j_-(j_>i_)
            if((not index_ == index) or (j_ == i_)):
                j_ = (index)%(nbBlocks)+1
                index_ = i_*(nbBlocks)+j_-(j_>i_)
            assert index_ == index
            return (i_,j_)
        else:
            assert (index>=0 and index == nbBlocks*(nbBlocks))
            return (0,0)

    def translate(action, nBlocks):
        if(action>=nBlocks*nBlocks):
            return (0,0)
        action_tuple = index_to_action_global(action, nBlocks)
        blocks = list(np.arange(nBlocks))
        blocks = [i+1 for i in blocks] + [0]
        a = 'move '
        a =(blocks[action_tuple[0]], blocks[action_tuple[1]])
        return a
    data = [translate(d, nBlocks) for d in data]
    return data

def data_to_string(data):
    s=''
    for i,d in enumerate(data):
        s+= str(d[0])+':'+str(d[1])
        if(i<len(data)-1):
            s+=','
    return s


def read_distribution(N_ACTIONS_TAKEN):
    directory = 'MiniCPBP/src/main/java/minicpbp/examples/data/ClassicalAIplanning/'
    filename  = 'cost_distribution.txt'
    data=''
    with open(directory+filename) as f:
        data = f.read()
    data_ = data.split('\n')
    data  = []
    for i in range(len(data_)):
        if(not data_[i]==''):
            data += [data_[i]]
    data = [str(int(d.split(':')[0])+N_ACTIONS_TAKEN)+":"+str(d.split(':')[1]) for d in data]
    #print(data)
    data = "#".join(data)
    #print('distribution:',data, type(data))
    return data
    
    #return float(expected_cost), int(minimum_cost), float(entropy)