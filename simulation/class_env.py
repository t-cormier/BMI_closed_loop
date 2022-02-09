import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import io


class Cortex(layers.Layer):
    def __init__(self, p):
        super(Cortex, self).__init__()

        # used params
        self.batch_size = p.batch_size
        self.input_gain = p.input_gain
        self.n_ctx = p.n_ctx
        self.n_str = p.n_str
        self.A_con = p.A_con
        self.B_con = p.B_con

        self.state_size = self.n_ctx

        self.disconnect_mask = tf.cast(np.diag(np.ones(self.n_ctx, dtype=np.bool)), tf.bool)
        self.sparse_mask = tf.random.uniform(shape=(self.n_ctx, self.n_ctx), seed = 0) <= self.A_con
        self.sparse_mask_B = tf.random.uniform(shape=(self.n_str, self.n_ctx), seed = 0) <= self.B_con

        initial_value_ext = tf.keras.initializers.Orthogonal(gain=p.B_init , seed=2)(shape=(self.n_str, self.n_ctx))
        self.input_weights = tf.Variable(initial_value=initial_value_ext,
                                         name='input_weights',
                                         trainable=False)

        initial_value_rec = tf.keras.initializers.Orthogonal(gain=p.A_init, seed=1)(shape=(self.n_ctx, self.n_ctx))
        self.recurrent_weights = tf.Variable(initial_value=initial_value_rec,
                                             name='recurrent_weights',
                                             trainable=False)


    def zero_state(self):
        x0 = 0.5 * tf.ones((self.batch_size, 1, self.n_ctx))
        return x0


    def sig(self, x):
        return 1/(1+tf.exp(-x))


    def call(self, inputs, state):
        x = state[0]

        # masks
        w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)
        w_rec = tf.where(self.sparse_mask, tf.zeros_like(w_rec), w_rec)
        w_in = tf.where(self.sparse_mask_B, tf.zeros_like(self.input_weights), self.input_weights)

        new_x = self.sig(tf.matmul(self.input_gain*inputs, w_in) + tf.matmul(x, w_rec))

        output = new_x
        return output, new_x




class Env(keras.Model):
    def __init__(self, p):
        super(Env, self).__init__()
        self.params = p
        self.n_step = p.n_step
        self.cell = Cortex(p)
        self.rnn = layers.RNN(self.cell, stateful=True, return_sequences=True)
        self.R1 = tf.Variable(p.R1, trainable=False)
        self.R2 = tf.Variable(p.R2, trainable=False)
        self.T = tf.Variable(p.T, trainable=False)

    def build(self):
        _ = self.rnn(tf.zeros(shape=(self.params.batch_size, 1, self.params.n_str)))


    def reward_policy(self, x):
        target = x[: ,:, self.R2.value()] - x[:, :, self.R1.value()]
        if self.params.shaping == None :
            cond = target >= self.T.value()
            reward = tf.cast(cond, tf.float32)
            done = tf.squeeze(cond)
        else :
            low_bn = (1 - self.params.shaping) * self.T.value()
            if target >= self.T.value():
                reward = tf.ones_like(target)
                done = tf.cast(True, tf.bool)
            elif (target < self.T.value()) and (target >= low_bn) :
                FACTOR = (target - low_bn) / (self.T.value() - low_bn) # in [0, 1]
                reward = FACTOR * tf.ones_like(target)
                done = tf.cast(False, tf.bool)
            else :
                reward = tf.zeros_like(target)
                done = tf.cast(False, tf.bool)
        return reward * self.params.r_scale, done


    def step(self, inputs, x_pre, x_bar_pre):
        x = self.rnn(inputs, initial_state=x_pre[0, :, :])
        # TODO: change so that it uses x_bar
        x_bar = self.params.x_decay * x_bar_pre + (1 - self.params.x_decay) * x
        reward, done = self.reward_policy(x_bar)
        return x, reward, done
