import numpy as np
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
    def __init__(self,nb_blocks,nb_actions_min,nb_actions_max, free_space=-1):
        super().__init__(nb_blocks*nb_blocks+2, nb_blocks+2+2, nb_actions_min, nb_actions_max)
        self.nb_blocks = nb_blocks
        self.init_config = list(np.zeros(nb_blocks, dtype=np.int64))
        self.goal_config = list(np.zeros(nb_blocks, dtype=np.int64))
        self.automatons  = list(np.zeros(nb_blocks, dtype=np.int64))
        self.free_space = free_space
        
    def create_automatons(self):
        """
        states : B_0..B_n-1, AIR, ACCEPT, REJECT
        actions: move(0,1), move(0,2).. move(0,n-1), move(1,0), move(1,2),..move(n,n-1), STAY
        """

        GROUND  = self.nb_blocks
                
        REJECT  = self.state_size - 1
        ACCEPT_finished  = self.state_size - 2
        ACCEPT_unfinished  = self.state_size - 3
        AIR     = self.state_size - 4
        
        states_accept = [ACCEPT_finished]
        
        #states_accept = list(np.arange(self.state_size)[:-1])
        #state_names[REJECT]='REJECT'
        #state_names[ACCEPT]='ACCEPT'
        #state_names[AIR]='AIR'
        for i in range(0, self.nb_blocks):
            #print(self.action_size)
            transition = np.zeros((self.state_size, (self.nb_blocks-1)*2 + 3), dtype=np.int64)
            cost_matrix= np.zeros((self.state_size, (self.nb_blocks-1)*2 + 3), dtype=np.int64)
            for s in range(transition.shape[0]):
                for a_index in range(transition.shape[1]):
                    next_state = None
                    action_local = index_to_action_local(a_index,i,self.nb_blocks)
                    """
                    (0,0) .. (n-2,0) place on top of k
                    (0,1) .. (n-2,1) place somewhere else
                    (0,2)            move k
                    (0,3)            STAY
                    """
                    if(s==ACCEPT_finished):
                        if(action_local[1]==4): 
                            next_state = ACCEPT_finished
                            cost_matrix[s][a_index] = 0
                        else:
                            next_state = REJECT
                            cost_matrix[s][a_index] = 0
                    elif(s==ACCEPT_unfinished):
                        if(action_local[1]==4): 
                            next_state = ACCEPT_unfinished
                            cost_matrix[s][a_index] = 1
                        else:
                            next_state = REJECT
                            cost_matrix[s][a_index] = 0
                    elif(s==REJECT):
                        next_state = REJECT
                        cost_matrix[s][a_index] = 0
                    else:
                        if(action_local[1]==4):
                            if(s==self.goal_config[i]):
                                next_state = ACCEPT_finished
                                cost_matrix[s][a_index] = 0#-self.nb_blocks
                            else:
                                next_state = ACCEPT_unfinished
                                cost_matrix[s][a_index] = 0#self.nb_blocks
                        elif(action_local[1]==3):
                            next_state = REJECT
                            #next_state = s
                            cost_matrix[s][a_index] = 1
                        else:
                            if(s==AIR):
                                if(action_local[1]==1 or action_local[1]==2):
                                    next_state = AIR
                                    cost_matrix[s][a_index] = 1
                                else:
                                    assert action_local[1]==0
                                    next_state = action_local[0]
                                    cost_matrix[s][a_index] = 1
                            else:
                                if(action_local[1]==0 or action_local[1]==2):
                                    next_state = REJECT
                                    cost_matrix[s][a_index] = 0
                                else:
                                    assert action_local[1]==1
                                    if(action_local[0]==s):
                                        next_state = AIR
                                        cost_matrix[s][a_index] = 1
                                    else:
                                        next_state = s
                                        cost_matrix[s][a_index] = 1
                    assert next_state>=0
                    transition[s][a_index]= next_state
            action_map = list(np.arange(self.action_size))
            #print('action map for block',i)
            
            for a_index in range(len(action_map)):
                a_global  = index_to_action_global(a_index, self.nb_blocks)
                a_local   = action_global_to_local(a_global[0],a_global[1],i)
                #print(a_global,'converted to',a_local,'(',action_to_index_local(a_local[0],a_local[1],i,self.nb_blocks),')')
                action_map[a_index] = action_to_index_local(a_local[0],a_local[1],i,self.nb_blocks)
            
            self.automatons[i] = Automaton(self.state_size            , 2*(self.nb_blocks-1)+3          , 
                                           self.init_config[i]        , states_accept, 
                                           action_map, cost_matrix,#list(np.ones(2*(self.nb_blocks-1)+1, dtype=np.int64))+[0],
                                           transition)
