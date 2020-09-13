model_fileName = "models/ppo1_8099328_steps.zip"

import gym, json, os, stable_baselines
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from env.ForexTradingEnv import ForexTradingEnv
from multiprocessing import freeze_support
import pandas as pd
from CustomPolicy import PPO2Policy_Basic

def main():
    agent_data = pd.read_csv('../output_EURUSD_M1_/agentData.csv')
    agent_data = agent_data.drop(agent_data.columns[0],axis=1)
    agent_data = agent_data.astype('float32')

    env = SubprocVecEnv([lambda: ForexTradingEnv(agent_data)]*10, )
    #env = DummyVecEnv([lambda: ForexTradingEnv(agent_data)], )

    #    model = DQN(CustomDQNPolicy, env, gamma=0.95, verbose=1, tensorboard_log = "./tensorboard", entcoeff=0.005, adam_epsilon = 1e-6)

    import tensorflow as tf
    from TenorboardCallbacks import TensorboardCallback
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./models/', name_prefix='ppo2')


    for curr in [1,2 , 3, 4, 5, 6, 7, 8, 9, 10]:
        model = PPO2(PPO2Policy_Basic, env, verbose=1, tensorboard_log = "./tensorboard", vf_coef = 1e-7, ent_coef=1e-4, n_steps=512, gamma=0.99)
        model.learn(total_timesteps=1000000000, log_interval=10000000, callback=CallbackList([TensorboardCallback(env), checkpoint_callback]))
        model.save(model_fileName)

    obs = env.reset()
    for i in range(2000000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if i % 1 == 0:
            env.render()
        if done:
            break


if __name__ == '__main__':
    freeze_support()
    main()
