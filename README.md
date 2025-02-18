### Semi-automatic reward shaping using constrained programming

Implementation of the reward shaping method as described in ```Shaping Reward Signals in Reinforcement Learning Using Constraint Programming``` (Accepted paper on CPAIOR 2025).

## Tutorial on how to run the code.
The ```nlm_training/neural-logic-machines``` folder contains code to train the neural logic machine to solve Blocks World 
problem instances. Each time a reward is required, it sends the needed information to a server to request the CPBP based reward.
- First, use ```export PATH=third_party/Jacinle_fresh/bin:$PATH``` to gain access to Google's Jacinle library. 
- Then, use ```jac-crun 0 scripts/blocksworld/learn_policy_cpbp.py --task final --use-gpu``` to start training.
- Before training, the two servers below must be running and listening also. Makes sure that the request is sent to the appropriate IP address/port ```NLM_untouched/neural-logic-machines/difflogic/envs/blocksworld/utils.py```

The 'listening_server' folder has a flask server which listens for reward requests. It takes the request information, processes
it, then sends it to a Java server.
- To run, use ```python server.py```, default port is 5000

The Java server in 'listening_server/MiniCPBP' is the constrained programming solver itself, when a reward is requested, it takes relevant information to 
create the model needed before computing and returning the reward.
- Run the ```AI_Front_train.java``` as the main file.
- Make sure to open the correct port.
