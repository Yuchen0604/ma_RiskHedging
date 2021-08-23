import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent

from env.FinalEnv import TrainEnv, TestEnv
import FinalAnalyzePlot

result_folder = "E:/final dqn/solar/ddqn/run7"  # modify required
weights_path = result_folder + "/weights.h5f"

train_file = result_folder + "/train_episode_R.txt"
test_income_file = result_folder + "/test_hour_income.txt"
test_action_file = result_folder + "/TestActions/test_action.csv"

# initialize training environment
training_data = pd.read_csv("./data/pv_2018.csv")  # modify required
train_env = TrainEnv(training_data)
# initialize testing env
testing_data = pd.read_csv("./data/pv_2019.csv")  # modify required
test_env = TestEnv(testing_data)

nb_actions = train_env.action_space.n

# build the neural network model
model = Sequential()
model.add(Flatten(input_shape=(1,) + train_env.observation_space.shape))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

memory = SequentialMemory(limit=50000, window_length=1)
training_policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1, value_min=.2,
                                       value_test=0, nb_steps=600000)
dqn = DQNAgent(model=model, nb_actions=nb_actions,
               memory=memory, policy=training_policy, target_model_update=500,
               enable_double_dqn=True)
dqn.compile(Adam(lr=.0005), metrics=['mae'])
dqn.fit(train_env, nb_steps=840000, visualize=True, verbose=2)
dqn.save_weights(weights_path, overwrite=True)

dqn.test(test_env, nb_episodes=1, visualize=True)

# plotting results
FinalAnalyzePlot.training_return(train_file)
FinalAnalyzePlot.training_profit(train_file)
FinalAnalyzePlot.training_pen(train_file)
FinalAnalyzePlot.training_cvar(train_file)
# FinalAnalyzePlot.training_negative(train_file)
FinalAnalyzePlot.cum_income(test_income_file)
FinalAnalyzePlot.battery_control(test_action_file)
FinalAnalyzePlot.income_to_cvar(test_income_file, 0.9)
FinalAnalyzePlot.plot_cvar(result_folder + "/cvar.csv")
FinalAnalyzePlot.storage_analyze(test_action_file)
