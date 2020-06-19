import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from math import copysign

#MAX_ACCOUNT_BALANCE = 2147483647
#MAX_NUM_SHARES = 2147483647
#MAX_OPEN_POSITIONS = 5
#MAX_STEPS = 20000

EMBEDDING_SIZE = 50
#BAR_HISTORY = 10

INITIAL_ACCOUNT_BALANCE = 1000
TARGET_ACCOUNT_BALANCE = 10*INITIAL_ACCOUNT_BALANCE
MIN_ALLOWED_POSITION = 50000
MAX_ALLOWED_POSITION = 100000
LEVERAGE = 100
COMMISSION = 0.00002

class ForexTradingEnv(gym.Env):
    """A forex trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, rates):
        super(ForexTradingEnv, self).__init__()

        self.rates = rates
        self.max_price = 2 #todo max

        # Actions: Buy, Sell, Pass
        self.action_space = spaces.Discrete(3)

        # Prices contains the OHCL values for the last five prices and last one is account info
        self.observation_space = spaces.Box(low=0, high=1, shape=(2+EMBEDDING_SIZE,), dtype=np.float16)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.position_vwap = 0
        self.position_volume = 0
        self.margin_used = 0
        self.equity = self.balance
        self.free_margin = self.equity
                
        # Set the current step to a random point within the data frame
        self.current_step = random.randint(0, self.rates.shape[0])

        return self._current_observation()


    def _current_observation(self):
        #frame = self.rates.loc[self.current_step - BAR_HISTORY: self.current_step-1, 'close.bid'].values / self.max_price
        frame = self.rates.iloc[self.current_step, -50:].values
        obs = np.append(frame, [max(0, self.equity / TARGET_ACCOUNT_BALANCE), (self.position_volume + MAX_ALLOWED_POSITION) /2/MAX_ALLOWED_POSITION] )
        return obs

    def _take_action(self, action):
        ask, bid = self.rates.loc[self.current_step-1, "close.ask"], self.rates.loc[self.current_step-1, "close.bid"]
        isDone = False

        # Buy minimum volume
        if action == 0:
            if self.position_volume < 0 :
                self.balance += (self.position_vwap - ask) * MIN_ALLOWED_POSITION - COMMISSION * MIN_ALLOWED_POSITION 
                self.position_volume += MIN_ALLOWED_POSITION
                self.margin_used = abs(self.position_volume) / LEVERAGE * ask
            elif self.free_margin >= MIN_ALLOWED_POSITION / LEVERAGE * ask: 
                if self.position_volume + MIN_ALLOWED_POSITION <= MAX_ALLOWED_POSITION:
                    self.position_vwap = (MIN_ALLOWED_POSITION * ask + self.position_vwap * abs(self.position_volume)) / (abs(self.position_volume) + MIN_ALLOWED_POSITION)
                    self.position_volume += MIN_ALLOWED_POSITION
                    self.margin_used =abs(self.position_volume) / LEVERAGE * ask
                    self.balance -= COMMISSION * MIN_ALLOWED_POSITION
            elif self.position_volume == 0: #we cant open any position now
                isDone = True
            else: #not enough margin to open new position
                self.balance *= 0.9

        # Sell minimum volume
        elif action == 1:
            if self.position_volume > 0 :
                self.balance += (bid - self.position_vwap) * MIN_ALLOWED_POSITION - COMMISSION * MIN_ALLOWED_POSITION 
                self.position_volume -= MIN_ALLOWED_POSITION
                self.margin_used = abs(self.position_volume) /LEVERAGE * ask
            elif self.free_margin > MIN_ALLOWED_POSITION / LEVERAGE * ask:
                if self.position_volume - MIN_ALLOWED_POSITION >= -MAX_ALLOWED_POSITION:
                    self.position_vwap = (MIN_ALLOWED_POSITION * bid + self.position_vwap * abs(self.position_volume)) / (abs(self.position_volume) + MIN_ALLOWED_POSITION)
                    self.position_volume -= MIN_ALLOWED_POSITION
                    self.margin_used = abs(self.position_volume) / LEVERAGE * ask
                    self.balance -= COMMISSION * MIN_ALLOWED_POSITION
            elif self.position_volume == 0:
                isDone = True   
            else: #not enough margin to open new position
                self.balance *= 0.9           
        
        #stopout test
        worst_equity = self._calculateEquity(self.rates.loc[self.current_step, "high.ask"], self.rates.loc[self.current_step, "low.bid"])
        if self.margin_used != 0 and worst_equity / self.margin_used < 0.5:
            self.balance = worst_equity
            self.position_volume = self.margin_used = 0
        #go to next state
        self.equity = self._calculateEquity(ask, bid)
        self.free_margin = self.equity - self.margin_used

        if self.equity > TARGET_ACCOUNT_BALANCE:
            isDone = True

        return isDone


    def _calculateEquity(self, ask, bid):
        price = bid if self.position_volume > 0 else ask
        return self.balance + (price - self.position_vwap) * abs(self.position_volume) * copysign(1, self.position_volume)      
        
    def step(self, action):
        self.lastAction = action
        prevEquity = self.equity
        # Execute one time step within the environment
        isDone = self._take_action(action)
        self.current_step += 1
        if self.current_step >= self.rates.shape[0]-2:
            isDone = True
        reward = (self.equity - prevEquity) / INITIAL_ACCOUNT_BALANCE 
        obs = self._current_observation()

        return obs, reward, isDone, {}

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Env[Step,Price] [{self.current_step},{self.rates.loc[self.current_step, "close.bid"]}] Agent[volume, equity] [{self.position_volume},{self.equity}]')

    def getLastAction(self):
        return self.lastAction