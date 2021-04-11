import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from env.S4A11R5Env import S4A11R5Env
from myrl.callbacks import TestCallback
from myrl.agents import OriginalDQNAgent

df = pd.read_csv('./data/da_2018_norm.csv')
# initialize the environment
env = S4A11R5Env(df)
nb_actions = env.action_space.n

# build the neural network model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(7))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# log all metrics to TensorBoard
tb_callback = TensorBoard(log_dir='./logs/s4a11r5/run3')
test_callback = TestCallback(tb_callback)
callbacks = [
    tb_callback,
    test_callback
]

memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy(eps=0.2)
# also tried LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.01, value_test=.05, nb_steps=100000)
dqn = OriginalDQNAgent(model=model, nb_actions=nb_actions, memory=memory, policy=policy, target_model_update=1e-2, gamma=0.999)
# other prams ever tried: target_model_update = 10000 (steps), gamma = 0.9
dqn.compile(Adam(lr=.008), metrics=['mae'])
# also tried lr = 0.01, 0.001, 0.0001
dqn.fit(env, nb_steps=840000, visualize=True, verbose=2, callbacks=callbacks) # main function of training

"""
This is the main process of dqn.fit function:
while step < nb_steps:
    call env.reset() to get initial state
    compute action in dqn.forward()
    call env.step() to execute action and get tuple(s',a,r,done,info)
    call dqn.backward() to train NN and get the metrics
"""