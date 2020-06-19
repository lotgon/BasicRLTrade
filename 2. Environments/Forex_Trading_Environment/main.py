model_fileName = "easy_ppo_model.zip"
learn_steps = 10000

import gym, json, os, stable_baselines
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1
from stable_baselines import DQN
from stable_baselines.deepq.policies import FeedForwardPolicy
import tensorflow as tf
from TenorboardCallbacks import TensorboardCallback

from env.ForexTradingEnv import ForexTradingEnv
import pandas as pd

agent_data = pd.read_csv('../output_EURUSD_M1/easyAgentData.csv')
agent_data = agent_data.drop(agent_data.columns[0],axis=1)
agent_data = agent_data.astype('float16')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: ForexTradingEnv(agent_data)], )

# Custom MLP policy of two layers of size 32 each
class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[512, 1024, 128, 32],
                                           layer_norm=False,
                                           feature_extraction="mlp")

if os.path.exists(model_fileName):
    model = DQN.load(model_fileName, env, tensorboard_log = "./tensorboard")
else:
    policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[512, 1024, 128, 32])
    model = DQN(CustomDQNPolicy, env, gamma=0.995, n_cpu_tf_sess=2 , verbose=1, tensorboard_log = "./tensorboard")#, policy_kwargs=policy_kwargs)

for i in range(learn_steps):
    model.learn(total_timesteps=10000000, log_interval=10000000, callback=TensorboardCallback(env))
    model.save(model_fileName)

obs = env.reset()
for i in range(2000000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if i % 100 == 0:
        env.render()
    if done:
        break

