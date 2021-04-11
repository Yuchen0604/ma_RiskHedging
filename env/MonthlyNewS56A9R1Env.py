import numpy as np
import pandas as pd
import random

import gym
from gym import spaces

U = 10
ETA_CH = 0.9
ETA_DIS = 0.9
SOC_MAX = 0.9
SOC_MIN = 0.1
R_CH = 3
R_DIS = 3
DAILY_COST = 20
FILE = "./data/results/reward_tmu7200.txt"

class DAR1Env(gym.Env):
    def __init__(self, data):
        self.df = data
        self.end_index = 4343
        #self.max_steps = len(self.df.loc[:, 'date'].values)
        self.ep_number = 0
        self.record_per_training = pd.DataFrame()
        file = open(FILE, "a+")
        file.write("ep,reward\n")
        file.close()

        high = np.finfo(np.float32).max
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-high, high=high, shape=(56,), dtype=np.float32)

    def _next_observation(self):
        # observation of hour t
        array1 = np.array([
            self.df.loc[self.current_step, 'norm_fg[MWh]'],
            self.df.loc[self.current_step, 'norm_yp[€/MWh]'],
            self.df.loc[self.current_step, 'norm_hh'],
            self.df.loc[self.current_step, 'norm_dd'],
            self.df.loc[self.current_step, 'norm_mm'],
        ], dtype=np.float32)
        # observation of day d
        start_pos = self.current_step // 24 * 24
        end_pos = start_pos + 23
        array2 = np.array([
            self.df.loc[start_pos: end_pos, 'norm_fg[MWh]'],
            self.df.loc[start_pos: end_pos, 'norm_yp[€/MWh]'],
        ])
        array2 = array2.reshape(48, )
        # state of charge of t-1, available upward energy and downward energy for hour t
        self.e_up = U * (SOC_MAX - self.soc)
        self.e_dn = U * (self.soc - SOC_MIN)
        e_up = self.e_up / 8
        e_dn = self.e_dn / 8
        array3 = np.array([self.soc, e_up, e_dn])

        obs = np.concatenate((array1, array2, array3))
        return obs

    def _take_action(self, action):
        self.action = action
        self.gen_fore = self.df.loc[self.current_step, 'fore_gen[MWh]']

        if action == 0:
            # no operation
            self.e_sell = self.gen_fore
            self.e_in = 0
            self.e_out = 0
        if 1 <= action <= 4:
            # charge 25%, 50%, 75%, or 100% of forecasted generation
            x = action / 4
            self.e_in = x * min(self.gen_fore*ETA_CH, R_CH, self.e_up)
            self.e_out = 0
            e_ch = self.e_in / ETA_CH
            self.e_sell = max(0, self.gen_fore - e_ch)
        elif 5 <= action <= 8:
            # discharge 25%, 50%, 75%, or 100% of stored electricity in the battery
            y = (action - 4) / 4
            self.e_out = y * min(R_DIS, self.e_dn)
            self.e_in = 0
            e_dis = self.e_out * ETA_DIS
            self.e_sell = self.gen_fore + e_dis
        self.soc = self.soc + (self.e_in - self.e_out) / U

    def step(self, action):
        # take action as input, return new state
        self.hour = self.df.loc[self.current_step, 'hour']
        self.day = self.df.loc[self.current_step, 'day']
        self.month = self.df.loc[self.current_step, 'month']
        dp = self.df.loc[self.current_step, 'day_price[€/MWh]']

        self._take_action(action)
        hourly_income = self.e_sell * dp
        self.hi_storage += hourly_income

        if self.hour == 23:
            self.daily_income = self.hi_storage
            self.hi_storage = 0
            daily_cost = DAILY_COST
            '''
            start = self.current_step - 23
            end = self.current_step
            self.dp_storage = self.df.loc[start: end, 'day_price[€/MWh]']
            self.daily_income = np.sum(self.sell_storage * self.dp_storage)
            self.sell_storage = np.zeros(24)
            self.dp_storage = np.zeros(24)
            '''
        else:
            daily_cost = 0
            self.daily_income = 0

        self.reward = self.daily_income - daily_cost
        self.cumulative_reward += self.reward

        self.current_step += 1
        self.done = bool(self.current_step == self.end_index)
        if self.done:
            self.current_step = self.end_index - 1
            next_obs = self._next_observation()
            self.ep_number += 1
        else:
            next_obs = self._next_observation()

        return next_obs, self.reward, self.done, {}

    def reset(self):
        # reset env, return initial state
        self.soc = random.randint(SOC_MIN * U, SOC_MAX * U) / U
        self.current_step = 3623
        self.hi_storage = 0
        self.reward = 0
        self.cumulative_reward = 0
        self.record_per_ep = pd.DataFrame()

        return self._next_observation()

    def render(self, mode='human', ):
        yp = self.df.loc[self.current_step, 'yes_price[€/MWh]']
        value1 = {
                 "month(t)": self.month,
                 "day(t)": self.day,
                 "hour(t)": self.hour,
                 "yesterday_price(t)": yp,
                 "soc(t)": self.soc,
                 "forecasted_generation(t)": self.gen_fore,
                 "action(t)": self.action,
                 "electricity_to_sell(t)": self.e_sell,
                 "e_in(t)": self.e_in,
                 "e_out(t)": self.e_out,
                 "e_up(t+1)": self.e_up,
                 "e_dn(t+1)": self.e_dn,
                 "daily_income": self.daily_income,
                 "reward_per_step": self.reward,
                 "cumulative_reward": self.cumulative_reward
                  }
        self.record_per_ep = self.record_per_ep.append(value1, ignore_index=True)
        if self.done:
            file = open(FILE, "a+")
            file.write(str(self.ep_number) + "," + str(self.cumulative_reward) + "\n")
            file.close()
            if self.ep_number % 10 == 0:
                self.record_per_ep.to_csv('./data/record/record_ep_{}.csv'.format(self.ep_number))
