import numpy as np
import pandas as pd

from env.NewS55A5R4Env import weeklyR4Env

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from myrl.agents import NewDQNAgent
from myrl.policy import NewEpsGreedyQPolicy
from rl.policy import LinearAnnealedPolicy
from rl.memory import SequentialMemory

from keras.callbacks import TensorBoard
from myrl.callbacks import TestCallback

df = pd.read_csv('./data/da_2018_norm.csv')
env = weeklyR4Env(df)

np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
# print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
policy = LinearAnnealedPolicy(NewEpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.01,
                              value_test=.05, nb_steps=100000)
tb_callback = TensorBoard(log_dir='./logs/weeklyR4')
test_callback = TestCallback(tb_callback)
callbacks = [
    tb_callback,
    test_callback
]
dqn = NewDQNAgent(model=model, nb_actions=nb_actions, memory=memory, policy=policy, target_model_update=10000)
dqn.compile(Adam(lr=.001), metrics=['mae'])
dqn.fit(env, nb_steps=1747200, visualize=True, verbose=2, callbacks=callbacks)
# hist = dqn.fit(env, nb_steps=1680, visualize=True, verbose=2, callbacks=[tensorboard])
#print(hist.history.keys())


