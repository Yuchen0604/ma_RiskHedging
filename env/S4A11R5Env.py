import numpy as np
import pandas as pd
import random
import gym
from gym import spaces

U = 5
ETA_CH = 0.9
ETA_DIS = 0.9
SOC_MAX = 0.9
SOC_MIN = 0.1
R_CH = 1
R_DIS = 1

# path to save the episode reward after whole training
FILE1 = "E:/training/s4a11r5/run3/results.txt"
# path to save variable values of every x episodes
FILE2 = "E:/training/s4a11r5/run3/record/record_ep_{}.csv"


# create custom gym environment for day-ahead market
class S4A11R5Env(gym.Env):

    def __init__(self, data):
        print("current training on: S4A11R5Env \n")
        # define input data
        self.df = data
        #  record the current training episode to save training results
        self.ep_number = 0

        file = open(FILE1, "a+")
        file.write("ep,index,reward\n")
        file.close()

        """
        Observations:
        type: gym.spaces.Box(n-dim real valued numbers of range [low, high])
            Observation                            Min                     Max
            state of charge(t)                      0                       1
            normalized hour identifier(t)           0                       1
            normalized forecasted generation(t)     0                       1
            normalized yesterday price(t)           0                       1
        Actions:
        type: Discrete(n discrete numbers of range [0, n-1])
            Num    Action
            0      neither charge nor discharge
            1-5    charge
            6-11   discharge
        """
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(11)

    def _next_observation(self):
        # return observation of hour t
        obs = np.array([
            self.soc,
            self.df.loc[self.current_index, 'norm_hh'],
            self.df.loc[self.current_index, 'norm_fg'],
            self.df.loc[self.current_index, 'norm_yp'],
        ], dtype=np.float32)

        return obs

    def step(self, action):
        """
        interaction between agent and environment
        :param action: chosen action from agent
        :return: a tuple (new state, reward, whether episode is done, information)
        """
        self._take_action(action)
        self.yp = self.df.loc[self.current_index, 'yes_price']  # denormalized
        self.unhedged_income = self.gen_fore * self.yp
        self.hedged_income = self.e_sell * self.yp
        self.reward = self.hedged_income - self.unhedged_income
        self.total_reward += self.reward
        """
        the cumulative reward of one episode indicates the profit/loss by performing charging/discharging
        profitable, when episode reward >0,
        no difference, when episode reward = 0,
        losing, when episode reward <0
        """

        self.nb_steps += 1  # number of steps within one episode
        self.done = bool(self.nb_steps == 168)
        if self.done:
            self.ep_number += 1

        self.current_index += 1  # index of csv file
        # calling the function with updated index to get state of next hour
        next_obs = self._next_observation()

        return next_obs, self.reward, self.done, {}

    def _take_action(self, action):
        """
        this function is called when agent interacts with env.
        it tells the agent what does each action number means
        """
        # calculate available upward and downward capacity with soc
        self.e_up = U * (SOC_MAX - self.soc)
        self.e_dn = U * (self.soc - SOC_MIN)
        # calculate selling e, charging/discharging e, and battery in/out e
        self.action = action
        self.gen_fore = self.df.loc[self.current_index, 'fore_gen']

        # no operation
        if action == 0:
            self.display_action = 0  # variable used to plot the result
            self.e_sell = self.gen_fore
            self.e_in = 0
            self.e_out = 0
        # charge 20%, 40%, 60%, 80% or 100% of C_max
        if 1 <= action <= 5:
            percent = action / 5
            self.display_action = -percent  # variable used to plot the result
            self.e_in = percent * min(self.gen_fore * ETA_CH, R_CH, self.e_up)
            self.e_out = 0
            e_ch = self.e_in / ETA_CH
            self.e_sell = max(0, self.gen_fore - e_ch)
        # charge 20%, 40%, 60%, 80% or 100% of D_max
        elif 6 <= action <= 10:
            percent = (action - 5) / 5
            self.display_action = percent  # variable used to plot the result
            self.e_out = percent * min(R_DIS, self.e_dn)
            self.e_in = 0
            e_dis = self.e_out * ETA_DIS
            self.e_sell = self.gen_fore + e_dis
        # update soc
        self.soc = self.soc + (self.e_in - self.e_out) / U

    def reset(self):
        # randomly choose one week to train
        indexes = []
        for i in range(0, 52):
            value = i * 168
            indexes.append(value)
        self.start_index = random.choice(indexes)
        self.current_index = self.start_index
        self.nb_steps = np.int(0)
        # initialize variables
        self.soc = SOC_MIN
        self.unhedged_income = np.float32(0)
        self.hedged_income = np.float32(0)
        self.reward = np.float32(0)
        self.total_reward = np.float32(0)
        self.record_per_ep = pd.DataFrame()

        return self._next_observation()

    def render(self, mode='human'):
        """
        this function is used to load the animated training process for env. such as Atari games
        it is used in this env. to save training results.
        """
        render_index = self.current_index - 1
        date = self.df.loc[render_index, 'Date']
        hour = self.df.loc[render_index, 'hour']
        weekday = self.df.loc[render_index, 'weekday']
        week = self.df.loc[render_index, 'week']
        norm_yp = self.df.loc[self.current_index, 'norm_yp']

        value1 = {
            "date": date,
            "hour": hour,
            "weekday": weekday,
            "week": week,
            "yp": self.yp,
            "norm yp": norm_yp,
            "forecasted_generation": self.gen_fore,
            "soc(t+1)": self.soc,
            "e_up": self.e_up,
            "e_dn": self.e_dn,
            "action": self.action,
            "percent": self.display_action,
            "electricity_to_sell": self.e_sell,
            "e_in": self.e_in,
            "e_out": self.e_out,
            "reward": self.reward,
            "unhedged income": self.unhedged_income,
            "hedged income": self.hedged_income
        }

        self.record_per_ep = self.record_per_ep.append(value1, ignore_index=True)
        if self.done:
            file = open(FILE1, "a+")
            file.write(str(self.ep_number) + "," +
                       str(self.start_index) + "," +
                       str(self.total_reward) + "\n")
            file.close()
            if self.ep_number % 10 == 0:
                self.record_per_ep.to_csv(FILE2.format(self.ep_number), index=None)

