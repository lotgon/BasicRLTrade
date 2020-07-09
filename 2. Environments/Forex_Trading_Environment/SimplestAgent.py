model_fileName = "./ppo1_5199616_steps.zip"
import gym, json, os, stable_baselines
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from env.ForexTradingEnv import ForexTradingEnv
import pandas as pd

agent_data = pd.read_csv('../output_EURUSD_M1_sin/agentData.csv')
agent_data = agent_data.drop(agent_data.columns[0],axis=1)
agent_data = agent_data.astype('float32')


def predict(obs):
    if obs[0] < 1.13 / 4 and obs[-1]<=0.5:
        return 2, 2
    if obs[0] > 1.14 / 4 and obs[-1]>=0.5:
        return 0, 0
    return 1, 1

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: ForexTradingEnv(agent_data)], )

obs = env.reset()
for i in range(2000000):
    action, _states = predict(obs[0])
    obs, rewards, done, info = env.step([action])
    if i % 100 == 0:
        env.render()
    if done:
        break


