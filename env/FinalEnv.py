import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
import os

U = 3  # battery capacity (MW)
ETA_CH = 0.9  # charging efficiency (%)
ETA_DIS = 0.9  # discharging efficiency (%)
SOC_MAX = 0.9  # maximum state of charge (%), the battery can only be charged to 5*0.9=4.5
SOC_MIN = 0.1  # minimum state of charge (%), the battery can only be discharged to 0.5
R_CH = .45  # maximum charging rate (MW), 0.15U
R_DIS = .45  # maximum discharging rate (MW), 0.15U
T = 168
CVAR_FACTOR = 0.5  # 0 for no CVAR term
BATTERY_COST = 10
PENALTY_FACTOR = 30

folder = "E:/final dqn/solar/ddqn/run7"  # modify required
# training file to draw the training curve
train_file = folder + "/train_episode_R.txt"
# testing files
# to draw cumulative reward
test_income_file = folder + "/test_hour_income.txt"
# to draw action
test_action_file = folder + "/TestActions/test_action.csv"


class TrainEnv(gym.Env):

    def __init__(self, data):
        os.mkdir(folder)
        os.mkdir(folder + "/TestActions")

        self.df = data
        self.max_start_index = self.df.shape[0] - T
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=-200, high=200, shape=(3,), dtype=np.float32)

        self.seed()

        self.ep_number = 0

        file = open(train_file, "a+")
        file.write("ep,profit,pen,cvar,return\n")
        file.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _next_observation(self):
        obs = np.array([
            self.soc,
            self.df.loc[self.current_index, 'fore_gen'],
            self.df.loc[self.current_index, 'fore_price'],
        ], dtype=np.float32)

        return obs

    def reset(self):
        self.start_index = np.random.randint(0, self.max_start_index)
        self.current_index = self.start_index
        self.soc = SOC_MIN
        self.total_reward = np.float32(0)
        self.total_profit = np.float32(0)
        self.total_penalty = np.float32(0)
        self.temp = []
        self.t = 0
        return self._next_observation()

    def step(self, action: int):
        # calculate available charging space and discharging capacity in battery
        e_up = U * (SOC_MAX - self.soc)
        e_dn = U * (self.soc - SOC_MIN)
        # calculate bidding e, charging/discharging e, and battery in/out e

        gen = self.df.loc[self.current_index, 'fore_gen']  # original forecasted data, not normalized
        act_price = self.df.loc[self.current_index, 'act_price']  # actual price

        # neither charge nor discharge, sell all of the generation
        e_in = 0
        e_out = 0
        e_sell = gen
        delta = 0
        reward = 0
        pen_ch = 0
        pen_dis = 0
        # charge battery at 20%, 40%, 60%, 80%, or 100% of C
        if 1 <= action <= 5:
            pct = action / 5
            e_in = min(R_CH, pct * min(gen * ETA_CH, e_up))
            e_ch = e_in / ETA_CH  # actual used generation to charge
            e_sell = max(0, gen - e_ch)
            delta = -e_ch
            if gen == 0 or self.soc == SOC_MAX:
                pen_ch = 1
        # discharge battery at 20%, 40%, 60%, 80%, or 100% of e_dn
        elif 6 <= action <= 10:
            pct = (action - 5) / 5
            e_out = min(R_DIS, pct * e_dn)
            e_dis = e_out * ETA_DIS
            e_sell = gen + e_dis
            delta = e_dis
            if self.soc == SOC_MIN:
                pen_dis = 1

        # update soc
        self.soc = self.soc + (e_in - e_out) / U
        # calculate reward,income
        reward += delta * act_price - BATTERY_COST * (e_in + e_out) - PENALTY_FACTOR * (pen_ch or pen_dis)
        self.total_reward += reward
        self.total_profit += delta * act_price - BATTERY_COST * (e_in + e_out)
        self.total_penalty += - PENALTY_FACTOR * (pen_ch or pen_dis)
        income = e_sell * act_price - BATTERY_COST * (e_in + e_out)
        self.temp.append(income)
        # move to next step
        self.current_index += 1
        next_obs = self._next_observation()

        self.done = bool((self.current_index - self.start_index) == T)  # one episode is one week (168 hours)
        if self.done:
            daily_icm = [sum(self.temp[i:i + 24]) for i in range(0, len(self.temp), 24)]
            avg = np.mean(daily_icm)
            dev = daily_icm - avg
            min_ndev = min(dev[dev < 0])
            assert min_ndev < 0
            cvar = CVAR_FACTOR * min_ndev
            reward += cvar
            self.total_reward += cvar

            self.ep_number += 1
            file = open(train_file, "a+")
            file.write(str(self.ep_number) + "," +
                       str(self.total_profit) + "," +
                       str(self.total_penalty) + "," +
                       str(cvar) + "," +
                       str(self.total_reward) + "\n")
            file.close()

        return next_obs, reward, self.done, {}

    def render(self, mode='human'):

        pass


class TestEnv(gym.Env):

    def __init__(self, data):
        self.df = data
        self.future_stp = 24
        self.max_index = self.df.shape[0] - self.future_stp
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=-200, high=200, shape=(3,), dtype=np.float32)

        file = open(test_income_file, "a+")
        file.write("month,hour,income_base,income_opt\n")
        file.close()

    def _next_observation(self):
        obs = np.array([
            self.soc,
            self.df.loc[self.current_index, 'fore_gen'],
            self.df.loc[self.current_index, 'fore_price'],
        ], dtype=np.float32)

        return obs

    def reset(self):
        self.current_index = 0
        self.soc = SOC_MIN
        self.record_per_ep = pd.DataFrame()
        self.pen_count = 0

        return self._next_observation()

    def step(self, action: int):
        # calculate available charging space and discharging capacity in battery
        e_up = U * (SOC_MAX - self.soc)
        e_dn = U * (self.soc - SOC_MIN)
        # calculate bidding e, charging/discharging e, and battery in/out e

        gen = self.df.loc[self.current_index, 'fore_gen']  # original forecasted data, not normalized
        act_price = self.df.loc[self.current_index, 'act_price']  # actual price

        # neither charge nor discharge, sell all of the generation
        e_in = 0
        e_out = 0
        e_ch = 0
        e_dis = 0
        e_sell = gen
        control = 0
        delta = 0
        reward = 0
        # charge battery at 20%, 40%, 60%, 80%, or 100% of C
        if 1 <= action <= 5:
            pct = action / 5
            e_in = min(R_CH, pct * min(gen * ETA_CH, e_up))
            e_ch = e_in / ETA_CH  # actual used generation to charge
            e_sell = max(0, gen - e_ch)
            control = -pct
            delta = -e_ch
            if gen == 0 or self.soc == SOC_MAX:
                self.pen_count += 1
        # discharge battery at 20%, 40%, 60%, 80%, or 100% of e_dn
        elif 6 <= action <= 10:
            pct = (action - 5) / 5
            e_out = min(R_DIS, pct * e_dn)
            e_dis = e_out * ETA_DIS
            e_sell = gen + e_dis
            control = pct
            delta = e_dis
            if self.soc == SOC_MIN:
                self.pen_count += 1

        # calculate income_base,income_opt,remainder
        income_base = gen * act_price
        income_opt = e_sell * act_price - (e_in + e_out) * BATTERY_COST
        # write to file
        month = self.df.loc[self.current_index, 'month']
        file = open(test_income_file, "a+")
        file.write(str(month) + "," +
                   str(self.current_index) + "," +
                   str(income_base) + "," +
                   str(income_opt) + "\n"
                  )
        file.close()
        # update soc
        self.soc = self.soc + (e_in - e_out) / U
        # log to record
        date = self.df.loc[self.current_index, 'Date']
        hour = self.df.loc[self.current_index, 'hour']
        map_ap = self.df.loc[self.current_index, 'map_ap']
        map_fp = self.df.loc[self.current_index, 'map_fp']

        value = {
            "date": date,
            "month": month,
            "hour": hour,
            "act_price": act_price,
            "map_ap": map_ap,
            "map_fp": map_fp,
            "fore_gen": gen,
            "soc": self.soc,
            "action": action,
            "e_ch": e_ch,
            "e_dis": e_dis,
            "delta": delta,
            "control": control,
            "e_sell": e_sell,
        }
        self.record_per_ep = self.record_per_ep.append(value, ignore_index=True)

        # move to next step
        self.current_index += 1
        next_obs = self._next_observation()

        # self.done = bool((self.current_index - self.start_index) == 168)  # one episode is one week (168 hours)
        self.done = bool(self.current_index == self.max_index)
        if self.done:
            self.record_per_ep.to_csv(test_action_file, index=None)
            print("total penalty:", self.pen_count)

        return next_obs, reward, self.done, {}

    def render(self, mode='human'):
        pass
