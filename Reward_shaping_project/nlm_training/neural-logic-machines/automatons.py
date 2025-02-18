import numpy as np


def action_to_index_global(i, j, nbBlocks):
    #returns the index for the action move(i,j) 
    if(i==0 and j==0):
        return nbBlocks*(nbBlocks)
    elif(i==1 and j==1):
        return nbBlocks*(nbBlocks)+1
    else:
        assert not i==j
        return i*(nbBlocks)+j-(j>i)
