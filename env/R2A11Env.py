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
R_CH = .5
R_DIS = .5
HOURLY_COST = 1
# reward_episode file
FILE1 = "E:/training/r2a11/run2/results.txt"
# result file
FILE2 = "E:/training/r2a11/run2/record/record_ep_{}.csv"


class R2A11Env(gym.Env):

    def __init__(self, data):
        print("current training on: R2A11Env. \n")
        self.df = data
        # self.start_index = -168
        self.ep_number = 0

        file = open(FILE1, "a+")
        file.write("ep,index,reward,income\n")
        file.close()

        high = np.finfo(np.float32).max
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=-high, high=high, shape=(51,), dtype=np.float32)

    def _next_observation(self):
        # observation of hour t
        hourly_obs = np.array([
            self.soc,
            self.df.loc[self.current_index, 'norm_fg'],
            self.df.loc[self.current_index, 'norm_yp'],
        ], dtype=np.float32)
        # observation of day d
        start_pos = self.current_index // 24 * 24
        end_pos = start_pos + 23
        array = np.array([
            self.df.loc[start_pos: end_pos, 'norm_fg'],
            self.df.loc[start_pos: end_pos, 'norm_yp'],
        ])
        daily_obs = array.reshape(48, )

        obs = np.concatenate((hourly_obs, daily_obs))
        return obs

    def _take_action(self, action):
        self.e_up = U * (SOC_MAX - self.soc)
        self.e_dn = U * (self.soc - SOC_MIN)
        # calculate bidding e, charging/discharging e, and battery in/out e
        self.action = action
        self.gen_fore = self.df.loc[self.current_index, 'fore_gen']
        if action == 0:
            # no operation
            self.percent = 0
            self.e_sell = self.gen_fore
            self.e_in = 0
            self.e_out = 0
        if 1 <= action <= 5:
            # charge 25%, 50%, 75%, or 100% of forecasted generation
            self.percent = action / 5
            self.e_in = self.percent * min(self.gen_fore * ETA_CH, R_CH, self.e_up)
            self.e_out = 0
            e_ch = self.e_in / ETA_CH
            self.e_sell = max(0, self.gen_fore - e_ch)
        elif 6 <= action <= 10:
            # discharge 25%, 50%, 75%, or 100% of stored electricity in the battery
            self.percent = (action - 5) / 5
            self.e_out = self.percent * min(R_DIS, self.e_dn)
            self.e_in = 0
            e_dis = self.e_out * ETA_DIS
            self.e_sell = self.gen_fore + e_dis

        self.soc = self.soc + (self.e_in - self.e_out) / U

    def step(self, action):
        # take action as input, calculate the reward, return new state
        self._take_action(action)
        self.yp = self.df.loc[self.current_index, 'yes_price']
        self.income = self.e_sell * self.yp
        self.reward = self.income - HOURLY_COST
        self.total_income += self.income
        self.total_reward += self.reward

        self.nb_steps += 1
        self.done = bool(self.nb_steps == 168)
        if self.done:
            self.ep_number += 1

        self.current_index += 1
        next_obs = self._next_observation()

        return next_obs, self.reward, self.done, {}

    def reset(self):
        # reset env, return initial state
        indexes = []
        for i in range(0, 52):
            value = i * 168
            indexes.append(value)
        self.start_index = random.choice(indexes)
        self.current_index = self.start_index
        print(self.current_index)
        self.nb_steps = np.int(0)
        """
        self.start_index += 168
        if self.start_index == 8736:
            self.start_index = 0
        self.current_index = self.start_index
        self.nb_steps = np.int(0)
        """
        self.soc = SOC_MIN
        self.income = np.float32(0)
        self.reward = np.float32(0)
        self.total_reward = np.float32(0)
        self.total_income = np.float32(0)
        self.record_per_ep = pd.DataFrame()

        return self._next_observation()

    def render(self, mode='human', ):
        render_index = self.current_index - 1
        date = self.df.loc[render_index, 'Date']
        hour = self.df.loc[render_index, 'hour']
        weekday = self.df.loc[render_index, 'weekday']
        week = self.df.loc[render_index, 'week']

        value1 = {
                 "date": date,
                 "hour": hour,
                 "weekday": weekday,
                 "week": week,
                 "yp": self.yp,
                 "forecasted_generation": self.gen_fore,
                 "soc(t+1)": self.soc,
                 "e_up": self.e_up,
                 "e_dn": self.e_dn,
                 "action": self.action,
                 "percent": self.percent,
                 "electricity_to_sell": self.e_sell,
                 "e_in": self.e_in,
                 "e_out": self.e_out,
                 "reward": self.reward,
                 "income": self.income,
                  }

        self.record_per_ep = self.record_per_ep.append(value1, ignore_index=True)
        if self.done:
            file = open(FILE1, "a+")
            file.write(str(self.ep_number) + "," +
                       str(self.start_index) + "," +
                       str(self.total_reward) + "," +
                       str(self.total_income) + "\n")
            file.close()
            if self.ep_number % 100 == 0:
                self.record_per_ep.to_csv(FILE2.format(self.ep_number), index=None)
