model_fileName = ""
learn_steps = 10000

import gym, json, os, stable_baselines
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1
#from stable_baselines.deepq.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

import tensorflow as tf
from TenorboardCallbacks import TensorboardCallback

from env.ForexTradingEnv import ForexTradingEnv
import pandas as pd

agent_data = pd.read_csv('../output_EURUSD_M1_sin/agentData.csv')
agent_data = agent_data.drop(agent_data.columns[0],axis=1)
agent_data = agent_data.astype('float32')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: ForexTradingEnv(agent_data)], )
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./models/', name_prefix='ppo1')

# Custom MLP policy of two layers of size 32 each
class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs, layers=[512, 1024, 512, 256, 256, 128, 128, 64, 32], layer_norm=True, feature_extraction="mlp")
class PPO1Policy_Basic(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        #super(PPO1Policy_Basic, self).__init__(*args, **kwargs, net_arch=[512, 1024, 512, 256, 256, 128, 128, 64, 32], feature_extraction="mlp")
        super(PPO1Policy_Basic, self).__init__(*args, **kwargs, net_arch=[32, 32, dict(pi=[32],vf=[64])], feature_extraction="mlp")

#if os.path.exists(model_fileName):
    #model = PPO1.load(model_fileName, env, tensorboard_log = "./tensorboard")
#else:
model = PPO1(PPO1Policy_Basic, env, gamma=0.97, verbose=1, tensorboard_log = "./tensorboard", entcoeff=0.01, adam_epsilon = 1e-5)
#    model = DQN(CustomDQNPolicy, env, gamma=0.95, verbose=1, tensorboard_log = "./tensorboard", entcoeff=0.005, adam_epsilon = 1e-6)

for i in range(learn_steps):
    model.learn(total_timesteps=10000000, log_interval=10000000, callback=CallbackList([TensorboardCallback(env), checkpoint_callback]))
    model.save(model_fileName)

obs = env.reset()
for i in range(2000000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if i % 100 == 0:
        env.render()
    if done:
        break

