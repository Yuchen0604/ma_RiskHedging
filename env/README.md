# custom OpenAI gym environment for day-ahead power market

A xxxEnv.py file specifies things below:
* how the state is like, including the initial state for the interaction
* what actions can agent choose, what does each action means
* which reward the agent can get after excute certain action
* how the loop of one episode is like 


## file name for env.

* MonthlyxxxxEnv: one episode is one month, others are trained with weekly data
* NewxxxxEnv: the Q value modification algorithm is used, files without "New" just use a normal Dqn agent
* SxAx: x is the number of dimentions for state and action
* Rx: x specify which kind of reward function is used

## different reward functions
![reward functions](/env/reward.jpg)

_file S4A11R5Env.py is with detailed annotation of the code_ 
 