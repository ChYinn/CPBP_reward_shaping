import numpy as np
#from flask import Flask, request as Flask, flask_request
import flask
import requests
from utils.server_utils import extract_config, create_file, solve_file, read_statistics, read_rewards
# create the Flask app
app = flask.Flask(__name__)
app.debug = False

@app.route('/solve_configuration', methods=['GET', 'POST'])
def query_example():
    content = flask.request.json
    config_initial = extract_config(content['config_initial'], int(content['nBlocks']))
    config_target  = extract_config(content['config_target'], int(content['nBlocks']))
    #print('actions raw:',content['actions'])
    mode = content['mode']
    if(content['isOver']=='True'):
        isOver  = True
    else:
        assert(content['isOver']=='False')
        isOver = False
    if(content['actions']==''):
        actions = []
    else:
        actions = [tuple([int(r) for r in v.replace("(","").replace(")","").split(",")]) for v in content['actions'].split("#")] #[int(a) for a in content['actions'].split(',')]
    if(content['action_distribution']==''):
        distribution = []
    else:
        distribution = [tuple([float(r) for r in v.replace("(","").replace(")","").split(",")]) for v in content['action_distribution'].split("#")]
    length_multiplier = float(content['length_multiplier'])
    thresh = float(content['threshold'])
    plan_size_try = int(content['plan_size_try'])
    """
    if(res=='Inconsistency error'):
        res = solve_file()
        print('Inconsistency error')
        print('Initial:',config_initial)
        print('Target:',config_target)
        print('Actions:',actions)"""
    tries = 0
    obtained = False
    inconsistency = 0
    aa = ''
    free_space = int(content['free_space'])
    max_stacks = int(content['max_stacks'])
    max_len    = int(content['max_len'])
    target_stacks = int(content['target_stacks'])
    #print('free_space:',free_space)
    distribution_data = 'nan'
    while(not obtained):
        actions_translated, planSize = create_file(int(content['nBlocks']), config_initial, config_target, actions, isOver, plan_size_try, length_multiplier=length_multiplier, free_space=free_space, max_stacks=max_stacks,target_stacks=target_stacks, max_len=max_len,thresh=thresh, distribution=distribution, model_mode=mode)
        res = solve_file()
        
        #print('[?]',res)
        #print('init:',config_initial,'target:',config_target,'actions:',actions,'nBlocks:',int(content['nBlocks']),'isOver:', isOver,'content[isOver]:', content['isOver'])# 'cost_exp:',str(expected_cost)[:4],'cost_min:',str(minimum_cost)[:4],'entropy:',str(entropy)[:4])
        N_ACTIONS_TAKEN = int(length_multiplier)
        #for i in range(N_ACTIONS_TAKEN):
        #    aa += '(x) '
        for i in range(len(actions)):
            aa += str(actions[i])+' '
        for i in range(planSize-len(actions)):
            aa += '(o) '
        
        aa += ', planSize: '+str(planSize)+' nActions:'+str(length_multiplier)+', init: '+str(config_initial)+', target: '+ str(config_target) + ', fs: '+str(free_space)
        
        if(res=='inconsistency'): #Remove and False if using Strict mode
            print('INCONSISTENCY!')
            expected_cost, minimum_cost, entropy = int(planSize), int(planSize), 0
            inconsistency = 1
            expected_cost = 2*int(content['nBlocks']) #+ N_ACTIONS_TAKEN#2*int(content['nBlocks'])#-planSize
            minimum_cost  = expected_cost
            distribution_data = str(expected_cost+N_ACTIONS_TAKEN)+":"+str(1)
            #print('[!] Inconsistency error')
        else:
            expected_cost, minimum_cost, entropy, distribution_data = read_rewards(config_initial, config_target, actions_translated, N_ACTIONS_TAKEN)
            inconsistency = 0
            
        #print('initial:',config_initial,'target:',config_target,'actions:',actions,'nBlocks:',int(content['nBlocks']),'isOver:', isOver,'content[isOver]:', content['isOver'], 'length_multiplier:',length_multiplier, 'cost_exp:',str(expected_cost)[:4],'cost_min:',str(minimum_cost)[:4],'java response:',res)
        
        
        obtained = expected_cost != None
        tries += 1
        if(tries>1):
            print("ERROR max tries exceeded")
            obtained = True
    """
    solve_file()
    sol = read_solution(int(content['nBlocks']))
    msg_sol = data_to_string(sol)
    s = read_statistics()
    """
    #IF S0 IS CURRENT STATE, MUST ADD TO COST PREVIOUS COSTS
    #expected_cost += length_multiplier
    #minimum_cost  += length_multiplier
    #OTHERWISE IGNORE
    expected_cost += 0
    minimum_cost  += 0
    
    if(isOver):
        expected_cost = 0#N_ACTIONS_TAKEN
        aa+=' [Done]'
    aa += '  E[C]: ' + str(expected_cost) +', E_min: '+ str(minimum_cost)
    if(inconsistency==1):
        aa+= ' **FAIL**'
    print(aa)
    s = str(expected_cost/(1))+','+str(minimum_cost/(1))+','+str(entropy/(1))+','+str(inconsistency)+','+distribution_data
    #s = str(expected_cost/(int(content['nBlocks'])))+','+str(minimum_cost/(int(content['nBlocks'])))+','+str(inconsistency)  
    #s = str(expected_cost/(int(content['nBlocks'])))+','+str(minimum_cost/(int(content['nBlocks'])))
    #print(s)
    #print(s)
    #print(config_initial)
    #print(config_target)
    #return msg_sol + "#" + s
    return s
@app.route('/form-example')
def form_example():
    return 'Form Data Example'

@app.route('/json-example')
def json_example():
    return 'JSON Object Example'

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(host='0.0.0.0',port=5000)
