import tensorflow as tf
from stable_baselines.common.callbacks import BaseCallback
import numpy as np

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, env, verbose=0):
        self.env = env
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log additional tensor
        #if not self.is_tb_set:
            #with self.model.graph.as_default():
                #tf.summary.scalar('value_target', tf.reduce_mean(self.model.value_target))
                #self.model.summary = tf.summary.merge_all()
            #self.is_tb_set = True
        # Log scalar value (here a random variable)
        value = self.env.env_method("getLastAction", indices=0)
        summary = tf.Summary(value=[tf.Summary.Value(tag='LastAction', simple_value=value[0])])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True