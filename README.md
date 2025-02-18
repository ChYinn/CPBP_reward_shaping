### Semi-automatic reward shaping using constrained programming
# The 'nlm_training/neural-logic-machines' folder contains code to train the neural logic machine to solve Blocks World 
problem instances. Each time a reward is required, it sends the needed information to a server to request the CPBP based reward.
## First, use $export PATH=third_party/Jacinle_fresh/bin:$PATH$
# The 'listening_server' folder has a flask server which listens for reward requests. It takes the request information, processes
it, then sends it to a Java server.
## To run, use python server.py
#The Java server in 'listening_server/MiniCPBP' is the constrained programming solver itself, when a reward is requested, it takes relevant information to 
create the model needed before computing and returning the reward.
## Run the AI_Front_train.java as the main file. 
