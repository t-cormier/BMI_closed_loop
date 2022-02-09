import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import io




# Create a layer for the environment

class Cortex(layers.Layer):
    def __init__(self, p):
        super(Cortex, self).__init__()

        self.params = p
        self.batch_size = p.batch_size
        self.n_ctx = p.n_ctx
        self.n_ext = p.n_ext

        self.state_size = self.n_ctx

        self.disconnect_mask = tf.Variable(tf.cast(np.diag(np.ones(self.n_ctx, dtype=np.bool)), tf.bool), trainable=False, name="disconnect_mask")
        self.sparse_mask = tf.Variable(tf.random.uniform(shape=(self.n_ctx, self.n_ctx)) <= self.params.p_con, trainable=False, name="sparse_mask")

        initial_value_ext = tf.keras.initializers.RandomNormal(seed=0)(shape=(self.n_ext, self.n_ctx))
        self.input_weights = tf.Variable(initial_value=initial_value_ext,
                                         name='input_weights',
                                         trainable=True)

        initial_value_rec = tf.keras.initializers.Orthogonal(gain=0.7, seed=0)(shape=(self.n_ctx, self.n_ctx))
        self.recurrent_weights = tf.Variable(initial_value=initial_value_rec,
                                             name='recurrent_weights',
                                             trainable=True)

        self.bias = tf.Variable(tf.zeros(shape=(1, self.n_ctx)),
                                name='bias_ctx',
                                trainable=True)

    def zero_state(self):
        x0 = tf.zeros((self.batch_size, 1, self.n_ctx))
        return x0

    def sig(self, x):
        return 1/(1+tf.exp(-x))

    def __call__(self, inputs, state, training=False):
        x = state[0]

        # masks
        w_rec = tf.where(self.disconnect_mask.value(), tf.zeros_like(self.recurrent_weights.value()), self.recurrent_weights.value())
        w_rec = tf.where(self.sparse_mask.value() , tf.zeros_like(w_rec), w_rec)

        new_x = self.sig(tf.matmul(inputs, self.input_weights.value()) + tf.matmul(x, w_rec) + self.bias.value())  # ideally (1, 1, n_ctx)

        output = new_x
        return output, new_x



class Env(keras.Model):
    def __init__(self, p):
        super(Env, self).__init__()
        self.cell = Cortex(p)
        self.rnn = layers.RNN(self.cell, stateful=True, return_sequences=True) # feed the RNN 1 step only at once


    def call(self, inputs):
        x = self.rnn(inputs)
        return x



# simulate with random input (generated as an initial continuous policy)
# record activity throughout the network
# visualize net act in 2D with umap
# determine a reward area and assess stability
