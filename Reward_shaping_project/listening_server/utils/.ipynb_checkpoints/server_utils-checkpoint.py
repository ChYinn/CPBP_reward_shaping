from automatons_test import BlocksWorld
import numpy as np
import requests

def extract_config(config_str, nBlocks):
    config_str = config_str.split(',')
    config = np.zeros(nBlocks, dtype=np.int64)
    for i in range(nBlocks):
        c_i = config_str[i].split(':')
        #print(c_i)
        config[int(c_i[0])] = int(c_i[1])
    return config

def create_file(nBlocks, init, target, actions, isOver, mode='train'):
    bw = BlocksWorld(nBlocks,nBlocks*4,nBlocks*4)
    AIR  = nBlocks
    STAY = nBlocks*nBlocks+1
    bw.init_config = init
    bw.goal_config = target
    bw.create_automatons()
    
    
    planSize = int(nBlocks*4)
    
    if(isOver):
        for i in range(len(actions),planSize):
            actions += [STAY]
    
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
        filename  = 'parameters.txt'
        with open(directory + filename, "w") as f:
            f.write(parameters_str)
    else:
        assert(mode=='solve','invalid mode!')
    
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
def read_rewards(init, target, actions):
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
        assert len(actions) == len(actions_read)
        for i in range(len(init)):
            assert(init_read[i]==init[i])
        for i in range(len(target)):
            assert(target_read[i]==target[i])
        for i in range(len(actions)):
            assert(actions_read[i]==actions[i])
    except:
        print('Read configuration mismatch, retrying...')
        print('init:',init,'read:',init_read)
        print('target:',target,'read:',target_read)
        print('actions:',actions,'read:',actions_read)
        return None, None
    expected_cost, minimum_cost = data[3+(len(actions_read)>0)].split(',')
    return float(expected_cost), int(minimum_cost)
    
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