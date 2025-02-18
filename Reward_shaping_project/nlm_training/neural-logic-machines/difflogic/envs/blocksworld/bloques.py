class bloque():
    def __init__(self, identity):
        self.above=None
        self.below=None
        self.identity = identity
        self.correctly_placed  = None
        self.correct_below      = None
        self.n_above = 0
    def check_correctly_placed(self):
        if(not self.correctly_placed==None):
            ...
        elif(self.below==None or self.correct_below==None):
            if(self.correct_below==None and self.below==None):
                self.correctly_placed = True
            else:
                self.correctly_placed = False
        else:
            self.correctly_placed = (self.below.identity == self.correct_below.identity) and (self.below.check_correctly_placed())
        return self.correctly_placed
    def set_above(self):
        if(self.above==None):
            self.n_above = 0
        else:
            self.n_above = self.above.set_above() + 1
        return self.n_above
    def get_n_below(self):
        if(self.below==None):
            return 0
        else:
            return self.below.get_n_below()+1
    
def get_wrong_blocks(init_config, target_config):
    bloques = [bloque(i) for i in range(len(init_config))]
    bloque_ground = bloque(len(init_config))
    for b in bloques:
        if(init_config[b.identity]<len(init_config)):
            b_above_init = bloques[init_config[b.identity]]
            b_above_init.below = b
            b.above = b_above_init
        if(target_config[b.identity]<len(target_config)):
            b_above_target = bloques[target_config[b.identity]]
            b_above_target.correct_below = b
    for b in bloques:
        b.check_correctly_placed()
    for b in bloques:
        b.set_above()
    for b in bloques:
        if(b.below==None):
            b.below = bloque_ground
            bloque_ground.n_above+=1
    wrong_bloques = []
    stacks = []
    for b in bloques:
        if(not b.correctly_placed):
            wrong_bloques+=[b]
    for i in range(len(init_config)):
        if(bloques[i].n_above==0):
            stacks += [bloques[i]]
    return wrong_bloques#, [b.n_above + b.correct_below]
def get_stacks(init_config, target_config):
    bloques = [bloque(i) for i in range(len(init_config))]
    bloque_ground = bloque(len(init_config))
    for b in bloques:
        if(init_config[b.identity]<len(init_config)):
            b_above_init = bloques[init_config[b.identity]]
            b_above_init.below = b
            b.above = b_above_init
        if(target_config[b.identity]<len(target_config)):
            b_above_target = bloques[target_config[b.identity]]
            b_above_target.correct_below = b
    for b in bloques:
        b.check_correctly_placed()
    for b in bloques:
        b.set_above()
    for b in bloques:
        if(b.below==None):
            b.below = bloque_ground
            bloque_ground.n_above+=1
    wrong_bloques = []
    stacks = []
    for b in bloques:
        if(not b.correctly_placed):
            wrong_bloques+=[b]
    for i in range(len(init_config)):
        if(bloques[i].n_above==0):
            stacks += [bloques[i]]
    return stacks#, [b.n_above + b.correct_below]
    