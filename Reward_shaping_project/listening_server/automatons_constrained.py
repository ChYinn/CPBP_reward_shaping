import numpy as np
import copy
import math
#Local actions (0,2) is ... (0,3) is stay 
def index_to_action_local(index,k,nbBlocks):
    """
    maps local index to the local action (u,v) for the block k
    """
    if(index<(nbBlocks-1)*2):
        i_ = int(index/(nbBlocks-1))
        j_ = index%(nbBlocks-1)           # (u,0) place u on  k
        index_ = i_*(nbBlocks-1)+j_-(j_>k)# (u,1) take  u off k
        if((not index_ == index) or (j_ == k)):
            j_ = (index)%(nbBlocks-1)+1
            index_ = i_*(nbBlocks-1)+j_-(j_>k)
        assert index_ == index
        return (j_,i_)
    elif(index==(nbBlocks-1)*2):          # place k somewhere
        return (0,2)  
    elif(index==(nbBlocks-1)*2+1):        # stay 
        return (0,3)
    else:
        assert index==(nbBlocks-1)*2+2    # finish
        return (0,4)
    

def action_to_index_local(u,v,k,nbBlocks):
    """
    maps local action (u,v) to the local index
    """
    if(v<2):
        return u +(nbBlocks-1)*v - (k<u)
    elif(v==2):
        return (nbBlocks-1)*2
    elif(v==3):
        return (nbBlocks-1)*2+1
    else:
        assert v==4
        return (nbBlocks-1)*2+2
        

def action_global_to_local(i,j,k):
    """
    maps a global action to the local action in the automaton for the block k
    
    (0,0) ... (n-2,0) place on top of k
    (0,1) ... (n-2,1) place somewhere else
    (0,2)             move k
    (0,3)             stay
    (0,4)             finish
    """
    u,v = None, None
    if(i==0 and j==0):
        u=0
        v=3
    elif(i==1 and j==1):
        u=0
        v=4
    elif(i==k):
        u=0
        v=2
    elif(j==k):
        u=i
        v=0
    else:
        u=i
        v=1
    return (u,v)
    

"""
def action_to_index_global(i, j, nbBlocks):
    #returns the index for the action move(i,j) 
    assert not i==j
    return i*(nbBlocks)+j-(j>i)
"""
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
        assert (index>=0)
        if(index == nbBlocks*(nbBlocks)):
            return (0,0)
        else:
            assert(index == nbBlocks*(nbBlocks)+1)
            return (1,1)

class AI_Plan:
    """
    action_size: number of types of actions 
    state_size : number of states B1,..Bn, AIR, ACCEPT, REJECT
    nb_actions_(min/max): lower and upper bounds for the number of actions needed
    """
    def __init__(self, action_size, state_size, nb_actions_min, nb_actions_max):
        self.action_size    = action_size
        self.state_size     = state_size
        self.nb_actions_min = nb_actions_min
        self.nb_actions_max = nb_actions_max
        self.automatons     = []

    def create_automatons(self):
        raise NotImplementedError
        
    def render(self):
        o=''
        o+= str(self.global_actions_keys['finish'])  + '\n'
        o+= str(self.nb_actions_min) +' '+ str(self.nb_actions_max) + ' \n'
        o+= str(self.action_size) + ' \n'
        o+= str(3) + ' 0 0 \n' 
        o+= str(len(self.automatons)) + '\n'
        
        for a in self.automatons:
            o+= '\n'
            o+= str(a.state_size) + ' \n'
            o+= str(a.init) + ' \n'
            o+= str(len(a.goal_states)) + '  '
            for s_goal in a.goal_states: 
                o += str(s_goal)+' '
            o+='\n'
            o+= str(a.action_size_local) + ' \n'
            for idx in range(len(a.action_map)):
                o+=str(a.action_map[idx]) + ' '
            o+='\n\n'
            for state in range(a.transition.shape[0]):
                for action in range(a.transition.shape[1]):
                    o+= str(a.transition[state][action]) +' '
                o+= '\n'
            o += '1\n'
            for state in range(a.cost_map.shape[0]):
                for action in range(a.cost_map.shape[1]):
                    o+= str(a.cost_map[state][action]) +' '
                o+= '\n'
        return o

class Automaton:
    """
    nb_states         : number of states of the automaton
    action_size_local : list of local actions
    goal_states       : list of accepting states
    action_map : F(X) -> Y | X : local action, Y : global action 
    cost_map   : C(X) -> Y | X : local action, Y : action cost
    """
    def __init__(self, state_size, action_size_local, init, goal_states, action_map, cost_map, transition):
        self.state_size         = state_size
        self.action_size_local = action_size_local
        self.goal_states       = goal_states
        self.action_map        = action_map
        self.cost_map          = cost_map
        self.init              = init
        self.transition        = transition
        
class BlocksWorld(AI_Plan):
    def __init__(self,nb_blocks, nb_actions_min, nb_actions_max):
        self.nb_blocks = nb_blocks
        self.global_actions = []
        self.global_actions_keys={}
        self.initiate_global_actions()
        states_0, keys_0 = self.get_local_states(0)
        #print(states_0)
        super().__init__(len(self.global_actions), len(states_0), nb_actions_min, nb_actions_max)
        self.init_config = list(np.zeros(nb_blocks, dtype=np.int64))
        self.goal_config = list(np.zeros(nb_blocks, dtype=np.int64))
        self.automatons  = list(np.zeros(nb_blocks, dtype=np.int64))
        #print('mid to expensive automaton')
        
    
    def initiate_global_actions(self):
        actions = []
        keys    = {}
        for i in range(self.nb_blocks+1):
            for j in range(self.nb_blocks+1):
                if(i==j or i==self.nb_blocks):
                    continue
                else:
                    j_ = j
                    if(j==self.nb_blocks):
                        j_='ground'
                    if(j_=='ground'):
                        keys[(i,j_,'added','final')] = len(actions)
                        actions += [(i,j_,'added','final')]
                        keys[(i,j_,'no_change','final')] = len(actions)
                        actions += [(i,j_,'no_change','final')]
                        
                        keys[(i,j_,'added','not_final')] = len(actions)
                        actions += [(i,j_,'added','not_final')]
                        keys[(i,j_,'no_change','not_final')] = len(actions)
                        actions += [(i,j_,'no_change','not_final')]
                        
                    else:
                        keys[(i,j_,'lifted','final')] = len(actions)
                        actions += [(i,j_,'lifted','final')]
                        keys[(i,j_,'no_change','final')] = len(actions)
                        actions += [(i,j_,'no_change','final')]
                        
                        keys[(i,j_,'lifted','not_final')] = len(actions)
                        actions += [(i,j_,'lifted','not_final')]
                        keys[(i,j_,'no_change','not_final')] = len(actions)
                        actions += [(i,j_,'no_change','not_final')]
                        
        for i in range(len(actions)):
            assert(keys[actions[i]] == i)
        #keys['stay'] = len(actions)
        #actions += ['stay']
        keys['finish'] = len(actions)
        actions += ['finish']
        self.global_actions      = actions
        self.global_actions_keys = keys
        
    def get_local_actions(self, block_i):
        actions_local       = copy.deepcopy(self.global_actions)
        actions_keys_local  = copy.deepcopy(self.global_actions_keys)
        
        
        actions_local = []
        actions_keys_local = {}
        map_global_to_local = list(np.zeros(len(self.global_actions)))
        for i in range(len(self.global_actions)):
            a_i_global = self.global_actions[i]
            if(a_i_global=='stay' or a_i_global=='finish'):
                map_global_to_local[i] = len(actions_local)
                actions_keys_local[a_i_global] = len(actions_local)
                actions_local += [a_i_global]
            else:
                if(not(a_i_global[0]==block_i or a_i_global[1]==block_i)):
                    if(not (a_i_global[0],'dummy','dummy','dummy') in actions_local):
                        map_global_to_local[i] = len(actions_local)
                        actions_keys_local[(a_i_global[0],'dummy','dummy','dummy')] = len(actions_local)
                        actions_local += [(a_i_global[0],'dummy','dummy','dummy')]
                    else:
                        map_global_to_local[i] = actions_keys_local[(a_i_global[0],'dummy','dummy','dummy')]
                else:
                    map_global_to_local[i] = len(actions_local)
                    actions_keys_local[a_i_global] = len(actions_local)
                    actions_local += [a_i_global]
            
        #print(map_global_to_local)
        #print(self.global_actions)
        #print(actions_local)
        #print('Block',block_i)
        #for i in range(len(self.global_actions)):
        #    print(self.global_actions[i],'->',actions_local[map_global_to_local[i]])
        return actions_local, actions_keys_local, map_global_to_local
    
    def get_action_map(self):
        return np.arange(len(self.global_actions), np.int64)
        
    def get_local_states(self, block):
        states_local = []
        states_keys_local={}
        for i in range(self.nb_blocks+1):
            '''
            for j in range(self.nb_blocks+1):
                if((i==j and (not (i==self.nb_blocks and j==self.nb_blocks))) or i==block or j==block):
                    continue
                else:
                    i_,j_ = i,j
                    if(i==self.nb_blocks):
                        i_= 'air'
                    if(j==self.nb_blocks):
                        j_ = 'ground'
                    states_keys_local[(i_,j_)] = len(states_local)
                    states_local += [(i_,j_)]
            '''
            if(i==block):
                continue
            i_ = i
            if(i==self.nb_blocks):
                i_= 'air'
            #states_keys_local[(i_,'ground')] = len(states_local)
            #states_local                    += [(i_,'ground')]
            
            #states_keys_local[(i_,'air')] = len(states_local)
            #states_local                    += [(i_,'air')]
            
            states_keys_local[(i_,'correct','is_ground')]     = len(states_local)
            states_local  += [(i_,'correct','is_ground')]
            states_keys_local[(i_,'correct','is_not_ground')] = len(states_local)
            states_local  += [(i_,'correct','is_not_ground')]

            
            states_keys_local[(i_,'incorrect','is_ground')]     = len(states_local)
            states_local  += [(i_,'incorrect','is_ground')]
            states_keys_local[(i_,'incorrect','is_not_ground')] = len(states_local)
            states_local  += [(i_,'incorrect','is_not_ground')]
            
            
        states_keys_local['accept_finished'] = len(states_local)
        states_local  += ['accept_finished']
        
        #states_keys_local['accept_unfinished'] = len(states_local)
        #states_local += ['accept_unfinished']
        states_keys_local['reject'] = len(states_local)
        states_local += ['reject']
        for i in range(len(states_local)):
            assert(states_keys_local[states_local[i]]== i)
        
        return states_local,states_keys_local
    
    def get_tuple_configuration(self, config):
        assert(len(config) == self.nb_blocks)
        tuple_config_np = np.ones((self.nb_blocks,2))*-1
        tuple_config    = []
        for i in range(self.nb_blocks):
            tuple_config_np[i,0]= config[i]
            if(not config[i] == self.nb_blocks):
                tuple_config_np[config[i],1] = i
        for i in range(self.nb_blocks):
            i_,j_ = int(tuple_config_np[i,0]), int(tuple_config_np[i,1])
            if(tuple_config_np[i,0]==self.nb_blocks):
                i_='air'
            if(tuple_config_np[i,1]==self.nb_blocks or tuple_config_np[i,1]==-1):
                j_='ground'
            tuple_config += [(i_,j_)]
            if(config[i]==self.nb_blocks):
                assert(tuple_config[i][0]=='air')
            else:
                assert(tuple_config[i][0]==config[i])
        return tuple_config
    def get_height(self, block, config):
        block_above = config[block]
        if(block_above==len(config)):
            return 1
        else:
            return 1+self.get_height(block_above, config)
    
    def get_config_height(self, config):
        heights = np.zeros(len(config), dtype=np.int64)
        for i in range(len(heights)):
            heights[i] = self.get_height(i,config)
        return heights
    
    def convert_to_local_actions(self, block):
        a = []
        
        ...
    
    def create_counter_automaton(self, m, max_stacks,target_stacks):
        states_counter = []
        states_counter_keys = {}
        for i in range(max_stacks):
            states_counter_keys[i] = len(states_counter)
            states_counter += [i]
        states_counter_keys['reject'] = len(states_counter)
        states_counter += ['reject']
        states_counter_keys['accept'] = len(states_counter)
        states_counter += ['accept']
        actions_counter = self.global_actions#      = actions
        actions_counter_keys = self.global_actions_keys# = keys
        
        transition = -1*np.ones((len(states_counter), len(actions_counter)), dtype=np.int64)
        cost_matrix= -1*np.ones((len(states_counter), len(actions_counter)), dtype=np.int64)
        states_accept = []
        for k in range(len(states_counter)):
            if(states_counter[k]=='accept'):
                states_accept += [states_counter_keys[states_counter[k]]]
        '''
        for k in range(len(states_counter)):
            if(not states_counter[k] == 'reject'):
                states_accept += [states_counter_keys[states_counter[k]]]
        '''
        for s_index in range(transition.shape[0]):
            for a_index in range(transition.shape[1]):
                next_state, next_cost = None,None
                action_local = actions_counter[a_index]
                state_local  = states_counter[s_index]
                if(state_local=='accept'):
                    if(action_local=='finish'): 
                        next_state = 'accept'
                        next_cost = 0
                    else:
                        next_state = 'reject'
                        next_cost = 0
                elif(state_local=='reject'):
                    next_state = 'reject'
                    next_cost  = 0
                elif((state_local == 0) and (action_local[2]=='added')):
                    next_state = 'reject'
                    next_cost  = 0
                elif((state_local == max_stacks-1) and (action_local[2]=='lifted')):
                    next_state = 'reject'
                    next_cost  = 0
                else:
                    if(action_local=='finish'):
                        if(state_local == (max_stacks-target_stacks)):
                            next_state = 'accept'
                            next_cost  = 0 
                        else:
                            next_state = 'reject'#'accept'#'reject'
                            next_cost  = 0 
                    elif(action_local[2]=='added'):
                        next_state = state_local - 1
                        next_cost  = 0
                    elif(action_local[2]=='lifted'):
                        next_state = state_local + 1
                        next_cost  = 0
                    elif(action_local[2]=='no_change'):
                        next_state = state_local
                        next_cost  = 0
                    else: 
                        assert(False)
                #print(next_state)
                transition[s_index][a_index]  = states_counter_keys[next_state]
                cost_matrix[s_index][a_index] = next_cost
                
        map_action = np.arange(len(actions_counter))
        self.automatons += [Automaton(len(states_counter)            , len(actions_counter)          , 
                                           states_counter_keys[m]        , states_accept, 
                                           map_action, cost_matrix, transition)]
        
        
    def create_automatons(self):
        init_config_tuple = self.get_tuple_configuration(self.init_config)
        goal_config_tuple = self.get_tuple_configuration(self.goal_config)
        
        goal_config_heights = self.get_config_height(self.goal_config)
        for i in range(0, self.nb_blocks):
            actions_i, actions_keys_i, map_global_to_local_i  = self.get_local_actions(i)
            states_i, states_keys_i    = self.get_local_states(i)
            states_accept = []
            '''
            for k in range(len(states_i)):
                if(not states_i[k] == 'reject'):
                    states_accept += [states_keys_i[states_i[k]]]
            '''
            
            for k in range(len(states_i)):
                if(states_i[k]=='accept_finished'):# or states_i[k]==(goal_config_tuple[i][0],'correct')):
                    states_accept += [states_keys_i[states_i[k]]]
                #if(states_i[k] == goal_config_tuple[i] or states_i[k]=='accept_finished'):
                #    states_accept += [states_keys_i[states_i[k]]]
            #print('states_accept:',states_accept)
            
            transition = -1*np.ones((self.state_size, len(actions_i)), dtype=np.int64)
            cost_matrix= -1*np.ones((self.state_size, len(actions_i)), dtype=np.int64)
            for s_index in range(transition.shape[0]):
                for a_index in range(transition.shape[1]):
                    next_state, next_cost = None,None
                    action_local = actions_i[a_index]
                    state_local  = states_i[s_index]
                    if(state_local=='accept_finished'):
                        if(action_local=='finish'): 
                            next_state = 'accept_finished'
                            next_cost = 0
                        else:
                            next_state = 'reject'
                            next_cost = 0
                    elif(state_local=='accept_unfinished'):
                        if(action_local=='finish'): 
                            next_state = 'accept_unfinished'
                            next_cost = 1#int(i==0)
                        else:
                            next_state = 'reject'
                            next_cost = 0
                    elif(state_local=='reject'):
                        next_state = 'reject'
                        next_cost = 0
                    else:
                        if(action_local=='finish'):
                            if((state_local[0], state_local[1])==(goal_config_tuple[i][0],'correct')):
                                next_state = 'accept_finished'
                                next_cost = 0#-self.nb_blocks#-goal_config_heights[i]#-(math.ceil(goal_config_heights[i]**0.75))#-(1)
                            else:
                                next_state = 'reject'#'accept_unfinished'
                                next_cost = 0 #self.nb_blocks
                            '''
                            else:
                                next_state = 'accept_finished'
                                next_cost = 2*self.nb_blocks
                            '''
                            
                        elif(action_local=='stay'):
                            next_state = 'reject'#state_local
                            next_cost = 0
                        elif(state_local[2]=='is_ground' and (action_local[0]==i and action_local[1]=='ground')):
                            next_state = 'reject'
                            next_cost  = 0
                        else:
                            if  (state_local[0]== 'air'):
                                if(action_local[0] == i):#if picking up this block
                                    if(state_local[1]=='correct' and False): #OPTINAL: remove useless actions
                                        next_state = 'reject'
                                        next_cost = 0
                                    elif(action_local[3]=='final' and (not action_local[1] == goal_config_tuple[i][1])): #if action has 'final' flag, but shouldn't
                                        next_state = 'reject'
                                        next_cost = 0
                                    elif(state_local[2]=='is_ground' and (not action_local[1]=='ground') and (action_local[2]=='lifted')):#lifting
                                        if(action_local[1]==goal_config_tuple[i][1]):
                                            if(action_local[3]=='final'):
                                                next_state = ('air','correct', 'is_not_ground')
                                                next_cost = 1
                                            else:
                                                next_state = 'reject'
                                                next_cost = 0
                                        else:
                                            next_state = ('air','incorrect','is_not_ground')
                                            next_cost = 1
                                    elif((state_local[2]=='is_not_ground') and (action_local[1]=='ground') and (action_local[2]=='added')):#adding
                                        if(action_local[1]==goal_config_tuple[i][1]):
                                            if(action_local[3]=='final'):
                                                next_state = ('air','correct',  'is_ground')
                                                next_cost = 1
                                            else:
                                                next_state = 'reject'
                                                next_cost = 0
                                        else:
                                            next_state = ('air','incorrect','is_ground')
                                            next_cost = 1
                                    #elif(((state_local[2]=='is_ground' and action_local[1]=='ground') or (state_local[2]=='is_not_ground' and (not action_local[1]=='ground'))) and (action_local[2]=='no_change')):# no_change
                                    elif((state_local[2]=='is_not_ground' and (not action_local[1]=='ground')) and (action_local[2]=='no_change')):
                                        if(action_local[1]==goal_config_tuple[i][1] and (action_local[3]=='final')):
                                            next_state = ('air','correct',  state_local[2])
                                            next_cost = 1
                                        else:
                                            next_state = ('air','incorrect',state_local[2])
                                            next_cost = 1
                                    else:
                                        next_state = 'reject'
                                        next_cost  = 0
                                elif(action_local[1]==i):
                                    if((not state_local[1] == 'correct') and action_local[3]=='final'):
                                        next_state = 'reject'
                                        next_cost = 0
                                    else:
                                        next_state = (action_local[0], state_local[1], state_local[2])
                                        next_cost  = 1
                                else:
                                    next_state = state_local
                                    next_cost = 1
                            else:
                                if(action_local[1]==i):
                                    '''
                                    if(action_local[0]==state_local[0]):
                                        next_state = (state_local[0], state_local[1])
                                        next_cost  = 0
                                    else:
                                    '''
                                    next_state = 'reject'
                                    next_cost   = 0
                                elif(action_local[0]==i):
                                    next_state = 'reject'
                                    next_cost = 0
                                elif(action_local[0]==state_local[0]):                                        #if removing the block above
                                    if(i==goal_config_tuple[state_local[0]][1] and state_local[1] == 'correct' and action_local[3]=='final'): #if the removed block was at the correct spot, it can't be final 
                                        next_state = 'reject'
                                        next_cost = 0
                                    else:
                                        next_state = ('air',state_local[1], state_local[2])
                                        next_cost = 1
                                else:
                                    next_state = state_local
                                    next_cost = 1
                    #assert next_state>=0
                    transition[s_index][a_index]  = states_keys_i[next_state]
                    cost_matrix[s_index][a_index] = next_cost
                    '''
                    try:
                        transition[s_index][a_index]  = states_keys_i[next_state]
                        cost_matrix[s_index][a_index] = next_cost
                    except:
                        print('action',action_local,'state', state_local,'next state', next_state,'block',i)
                    '''
            action_map = list(np.arange(self.action_size))
            #print(cost_matrix.shape)
            #print(cost_matrix)
            #print('action map for block',i)
            '''
            for a_index in range(len(action_map)):
                a_global  = index_to_action_global(a_index, self.nb_blocks)
                a_local   = action_global_to_local(a_global[0],a_global[1],i)
                #print(a_global,'converted to',a_local,'(',action_to_index_local(a_local[0],a_local[1],i,self.nb_blocks),')')
                action_map[a_index] = action_to_index_local(a_local[0],a_local[1],i,self.nb_blocks)
            '''
            
            init_i = (init_config_tuple[i][0],-1)
            ground_status_i = '' 
            
            def is_correct_spot(l, init_config_tuple, goal_config_tuple):
                if(init_config_tuple[l][1]==goal_config_tuple[l][1]):
                    if(init_config_tuple[l][1] == 'ground'):
                        return True
                    else: 
                        return is_correct_spot(init_config_tuple[l][1],init_config_tuple,goal_config_tuple)
                return False
            if(init_config_tuple[i][1] == 'ground'):
                ground_status_i = 'is_ground'
            else:
                ground_status_i = 'is_not_ground'
            correct_i = ''
            if(is_correct_spot(i, init_config_tuple, goal_config_tuple)):
                correct_i = 'correct'
            else:
                correct_i = 'incorrect'
            init_i = (init_config_tuple[i][0], correct_i, ground_status_i)
            '''
            if(init_config_tuple[i][1]==goal_config_tuple[i][1]):
                init_i = (init_config_tuple[i][0], 'correct', ground_status_i)
            else:
                init_i = (init_config_tuple[i][0], 'incorrect', ground_status_i)
            '''
            self.automatons[i] = Automaton(self.state_size            , len(actions_i)          , 
                                           states_keys_i[init_i]        , states_accept, 
                                           map_global_to_local_i, cost_matrix,#list(np.ones(2*(self.nb_blocks-1)+1, dtype=np.int64))+[0],
                                           transition)
            