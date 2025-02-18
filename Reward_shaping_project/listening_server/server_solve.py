
import numpy as np
#from flask import Flask, request as Flask, flask_request
import flask
import requests
from utils.server_utils import extract_config, create_file, solve_file, read_statistics, read_rewards, read_solution, data_to_string
# create the Flask app
app = flask.Flask(__name__)
app.debug = False

@app.route('/solve_configuration', methods=['GET', 'POST'])
def query_example():
    content = flask.request.json
    config_initial = extract_config(content['config_initial'], int(content['nBlocks']))
    config_target  = extract_config(content['config_target'], int(content['nBlocks']))
    #print(content['actions'])
    '''
    if(content['isOver']=='True'):
        isOver  = True
    else:
        assert(content['isOver']=='False')
        isOver = False
    if(content['actions']==''):
        actions = []
    else:
        actions = [int(a) for a in content['actions'].split(',')]

    length_multiplier = float(content['length_multiplier'])
    '''
    """
    if(res=='Inconsistency error'):
        res = solve_file()
        print('Inconsistency error')
        print('Initial:',config_initial)
        print('Target:',config_target)
        print('Actions:',actions)"""
    tries = 0
    obtained = False
    '''
    while(not obtained):
        create_file(int(content['nBlocks']), config_initial, config_target, actions, isOver,length_multiplier=length_multiplier)
        res = solve_file()
        #print('java response:',res)
        expected_cost, minimum_cost = read_rewards(config_initial, config_target, actions)
        #print('initial:',config_initial,'target:',config_target,'actions:',actions,'nBlocks:',int(content['nBlocks']),'isOver:', isOver,'content[isOver]:', content['isOver'], 'length_multiplier:',length_multiplier, 'cost_exp:',str(expected_cost)[:4],'cost_min:',str(minimum_cost)[:4],'java response:',res)
        print('init:',config_initial,'target:',config_target,'actions:',actions,'nBlocks:',int(content['nBlocks']),'isOver:', isOver,'content[isOver]:', content['isOver'], 'cost_exp:',str(expected_cost)[:4],'cost_min:',str(minimum_cost)[:4])
        obtained = expected_cost != None
        tries += 1
        if(tries>500):
            print("ERROR max tries exceeded")
            obtained = True
    '''
    create_file(int(content['nBlocks']), config_initial, config_target, [], False, length_multiplier=2)
    solve_file()
    sol = read_solution(int(content['nBlocks']))
    print(sol)
    msg_sol = data_to_string(sol)
    s = read_statistics()
    
    #s = str(expected_cost)+','+str(minimum_cost)
    #s = str(expected_cost/(int(content['nBlocks'])))+','+str(minimum_cost/(int(content['nBlocks'])))
    #print(s)
    #print(s)
    #print(config_initial)
    #print(config_target)
    return msg_sol + "#" + s
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
