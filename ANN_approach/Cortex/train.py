import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import io

import network_RNN as n

class Params:
    """store params for the simulation"""

    def __init__(self):
        super(Params, self).__init__()

        self.batch_size = 1
        self.n_ctx = 100
        self.n_ext = 20
        self.p_con = 0.1

        self.sim_len = 1000



class Data:
    """ store the data to be saved and processed post simulation (numpy)"""

    def __init__(self, p):
        self.x_hist = np.empty(shape=(p.sim_len, p.n_ctx), dtype=np.float32)

    def add_x(self, x_t, t):
        self.x_hist[t] = tf.squeeze(x_t).numpy() # x has to be a tensor

    def save(self, filename):
        np.save(f'{filename}.npy', self.x_hist)
        return 0


# simulate with random input (generated as an initial continuous policy)
# record activity throughout the network
# visualize net act in 2D with umap
# determine a reward area and assess stability
p = Params()
random_input = tf.random.normal(shape=(p.batch_size, p.sim_len, 1, p.n_ext))
env = n.Env(p)
data = Data(p)


for t in range(p.sim_len):
    input = random_input[:,t,:,:]
    x_t = env(input)
    data.add_x(x_t, t)

data.save('test')
