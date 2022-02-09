import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import io
import tqdm




class Params(object):
    """Params of the simulation"""

    def __init__(self):

        # sim params
        self.sim_ID = '62_TESTlearning'
        self.batch_size = 1
        self.n_episode = 1000
        self.t_limit = 1000  #ms (1 second)
        self.action_latency = 1 #ms
        self.monitor_elig_grads = False
        self.train_value_only = False
        self.value_net = 'SFB' # -opt = 'FFD', 'SFB'
        self.n_step = 1
        self.data_store = 'GPU' # -opt = 'GPU', 'CPU'
        self.l_baseline = 20
        self.test_learning = True


        #Env params
        self.n_ctx = 1000
        self.n_monitored = 1000
        idx = tf.random.shuffle(tf.range(self.n_ctx), seed=1)[:self.n_monitored]
        self.R1 = idx[0]
        self.R2 = idx[1]
        self.monitored_neurons = idx
        self.A_con = 0.4
        self.B_init = 0.5
        self.A_init = 0.5
        self.B_con = 0.5
        self.T = 0.
        self.shaping = None # -opt = ]0., 1], None     (shaping = None means no reward shaping)
        self.r_scale = 0.1
        self.input_gain = 3.

        #Agent params
        self.tau_sigma = 300 #seconds
        self.sigma_init = 1.
        self.learning_rate = 2e-3
        self.n_str = 100
        self.tau_gamma = 20 #ms
        self.gamma = tf.exp(-1/self.tau_gamma)
        self.css_exc = False
        self.w_init_exc = tf.math.sqrt(6/self.n_ctx)
        self.w_init = 1.
        self.random_bias_gain = 1.
        self.tau_x = 15 # ms
        self.x_decay = tf.exp(-1/self.tau_x)
        self.value_gain = 1.




class DataGPU:
    """DataGPU : tensorarray version of dataclass"""

    def __init__(self, params):
        super(DataGPU, self).__init__()
        self.params = params.__dict__

        self.R = tf.TensorArray(dtype=tf.int16, size=0, dynamic_size=True)
        self.R = self.R.write(0, tf.cast(params.R1, tf.int16))
        self.R = self.R.write(1, tf.cast(params.R2, tf.int16))

        # Updated every time step
        self.r_sim = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
        self.x_sim = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
        self.x_bar_sim = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
        self.f_sim = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
        self.value_sim = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
        self.mu_sim = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
        self.td_sim = tf.TensorArray(dtype=tf.float16, size=params.n_episode)

        # Updated every episode
        self.sigma_sim = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
        self.t_episode =  tf.TensorArray(dtype=tf.int16, size=params.n_episode)
        self.w_mu_sim = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
        self.b_mu_sim = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
        #self.w_value_sim = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
        #self.b_value_sim = tf.TensorArray(dtype=tf.float16, size=params.n_episode)

        if params.monitor_elig_grads :
            # RL elig traces
            self.w_mu_elig = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
            self.b_mu_elig = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
            #self.w_v_elig = tf.TensorArray(dtype=tf.float16, size=params.n_episode)

            # Grads
            self.d_dw_mu = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
            self.d_db_mu = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
            #self.d_dw_v = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
            #self.d_db_v = tf.TensorArray(dtype=tf.float16, size=params.n_episode)


    def update(self,
               x_ep,
               x_bar_ep,
               f_ep,
               v_ep,
               mu_ep,
               sigma_ep,
               td_ep,
               reward_ep,
               t_ep,
               w_mu,
               b_mu,
               #w_value,
               #b_value,
               w_mu_elig_ep=None,
               b_mu_elig_ep=None,
               #w_v_elig_ep=None,
               d_dw_mu_ep=None,
               d_db_mu_ep=None,
               #d_dw_v_ep=None,
               #d_db_v_ep=None,
               episode=0,
               dtype=tf.float16):
        self.r_sim = self.r_sim.write(episode, tf.cast(reward_ep, dtype))
        self.x_sim = self.x_sim.write(episode, tf.cast(x_ep, dtype))
        self.x_bar_sim = self.x_bar_sim.write(episode, tf.cast(x_bar_ep, dtype))
        self.f_sim = self.f_sim.write(episode, tf.cast(f_ep, dtype))
        self.value_sim = self.value_sim.write(episode, tf.cast(v_ep, dtype))
        self.mu_sim = self.mu_sim.write(episode, tf.cast(mu_ep, dtype))
        self.sigma_sim = self.sigma_sim.write(episode, tf.cast(sigma_ep, dtype))
        self.td_sim = self.td_sim.write(episode, tf.cast(td_ep, dtype))
        self.t_episode = self.t_episode.write(episode, tf.cast(t_ep, tf.int16))
        self.w_mu_sim = self.w_mu_sim.write(episode, tf.cast(w_mu, dtype))
        self.b_mu_sim = self.b_mu_sim.write(episode, tf.cast(b_mu, dtype))
        #self.w_value_sim = self.w_value_sim.write(episode, tf.cast(w_value, dtype))
        #self.b_value_sim = self.b_value_sim.write(episode, tf.cast(b_value, dtype))
        if self.params['monitor_elig_grads']:
            self.w_mu_elig = self.w_mu_elig.write(episode, tf.cast(w_mu_elig_ep, dtype))
            self.b_mu_elig = self.b_mu_elig.write(episode, tf.cast(b_mu_elig_ep, dtype))
            #self.w_v_elig = self.w_v_elig.write(episode, tf.cast(w_v_elig_ep, dtype))
            self.d_dw_mu = self.d_dw_mu.write(episode, tf.cast(d_dw_mu_ep, dtype))
            self.d_db_mu = self.d_db_mu.write(episode, tf.cast(d_db_mu_ep, dtype))
            #self.d_dw_v = self.d_dw_v.write(episode, tf.cast(d_dw_v_ep, dtype))
            #self.d_db_v = self.d_db_v.write(episode, tf.cast(d_db_v_ep, dtype))

    def stack(self):
        self.R = self.R.stack()
        self.r_sim = self.r_sim.stack()
        self.x_sim = self.x_sim.stack()
        self.x_bar_sim = self.x_bar_sim.stack()
        self.f_sim = self.f_sim.stack()
        self.value_sim = self.value_sim.stack()
        self.mu_sim = self.mu_sim.stack()
        self.sigma_sim = self.sigma_sim.stack()
        self.td_sim = self.td_sim.stack()
        self.t_episode = self.t_episode.stack()
        self.w_mu_sim = self.w_mu_sim.stack()
        self.b_mu_sim = self.b_mu_sim.stack()
        #self.w_value_sim = self.w_value_sim.stack()
        #self.b_value_sim = self.b_value_sim.stack()
        if self.params['monitor_elig_grads']:
            self.w_mu_elig = self.w_mu_elig.stack()
            self.b_mu_elig = self.b_mu_elig.stack()
            #self.w_v_elig = self.w_v_elig.stack()
            self.d_dw_mu = self.d_dw_mu.stack()
            self.d_db_mu = self.d_db_mu.stack()
            #self.d_dw_v = self.d_dw_v.stack()
            #self.d_db_v = self.d_db_v.stack()

    def save(self, filename):
        self.stack()
        dict = self.__dict__
        if not os.path.isdir("data/"):
            os.mkdir("data")
        np.save(f'data/{filename}.npy', dict)



# class DataCPU:
#     """DataCPU : nparray version of dataclass"""
#
#     def __init__(self, params):
#         super(DataCPU, self).__init__()
#         self.params = params.__dict__
#
#         self.R = np.zeros((2,), dtype=np.float16)
#
#         # Updated every time step
#         self.r_sim = np.zeros(shape=(params.n_episode, params.batch_size, params.t_limit), dtype=np.float16)
#         self.x_sim = np.zeros(shape=(params.n_episode, params.batch_size, params.t_limit, params.n_ctx), dtype=np.float16)
#         self.f_sim = np.zeros(shape=(params.n_episode, params.batch_size, params.t_limit, params.n_str), dtype=np.float16)
#         self.value_sim = np.zeros(shape=(params.n_episode, params.batch_size, params.t_limit), dtype=np.float16)
#         self.mu_sim = np.zeros(shape=(params.n_episode, params.batch_size, params.t_limit, params.n_str), dtype=np.float16)
#         self.td_sim = np.zeros(shape=(params.n_episode, params.batch_size, params.t_limit), dtype=np.float16)
#
#         # Updated every episode
#         self.sigma_sim = np.zeros(shape=(params.n_episode, params.batch_size, 1), dtype=np.float16)
#         self.t_episode = np.zeros(shape=(params.n_episode, params.batch_size, 1), dtype=np.float16)
#         self.w_mu_sim = np.zeros(shape=(params.n_episode, params.batch_size, params.n_ctx, params.n_str)), dtype=np.float16)
#         self.b_mu_sim = np.zeros(shape=(params.n_episode, params.batch_size, params.n_str, 1)), dtype=np.float16)
#
#         if params.monitor_elig_grads :
#             # RL elig traces
#             self.w_mu_elig = np.zeros(shape=(params.n_episode, params.batch_size, params.t_limit, params.n_ctx, params.n_str)), dtype=np.float16)
#             self.b_mu_elig = np.zeros(shape=(params.n_episode, params.batch_size, params.t_limit, params.n_str, 1)), dtype=np.float16)
#             #self.w_v_elig = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
#
#             # Grads
#             self.d_dw_mu = np.zeros(shape=(params.n_episode, params.batch_size, params.t_limit, params.n_ctx, params.n_str)), dtype=np.float16)
#             self.d_db_mu = np.zeros(shape=(params.n_episode, params.batch_size, params.t_limit, params.n_str, 1)), dtype=np.float16)
#             #self.d_dw_v = tf.TensorArray(dtype=tf.float16, size=params.n_episode)
#             #self.d_db_v = tf.TensorArray(dtype=tf.float16, size=params.n_episode)

    # def load_chosen_neurons(self, R1, R2, dtype=tf.int16):
    #     self.R[0] = tf.cast(R1, dtype)).numpy()
    #     self.R[1] = tf.cast(R2, dtype)).numpy()
    #
    # def update(self,
    #            x_ep,
    #            x_bar_ep,
    #            f_ep,
    #            v_ep,
    #            mu_ep,
    #            sigma_ep,
    #            td_ep,
    #            reward_ep,
    #            t_ep,
    #            w_mu,
    #            b_mu,
    #            #w_value,
    #            #b_value,
    #            w_mu_elig_ep=None,
    #            b_mu_elig_ep=None,
    #            #w_v_elig_ep=None,
    #            d_dw_mu_ep=None,
    #            d_db_mu_ep=None,
    #            #d_dw_v_ep=None,
    #            #d_db_v_ep=None,
    #            episode=0,
    #            dtype=tf.float16):
    #     self.r_sim[episode, :, :] += tf.cast(reward_ep, dtype)).numpy()
    #     self.x_sim[episode, :, :, :] = tf.cast(x_ep, dtype).numpy()
    #     self.x_bar_sim = self.x_bar_sim.write(episode, tf.cast(x_bar_ep, dtype))
    #     self.f_sim = self.f_sim.write(episode, tf.cast(f_ep, dtype))
    #     self.value_sim = self.value_sim.write(episode, tf.cast(v_ep, dtype))
    #     self.mu_sim = self.mu_sim.write(episode, tf.cast(mu_ep, dtype))
    #     self.sigma_sim = self.sigma_sim.write(episode, tf.cast(sigma_ep, dtype))
    #     self.td_sim = self.td_sim.write(episode, tf.cast(td_ep, dtype))
    #     self.t_episode = self.t_episode.write(episode, tf.cast(t_ep, tf.int16))
    #     self.w_mu_sim = self.w_mu_sim.write(episode, tf.cast(w_mu, dtype))
    #     self.b_mu_sim = self.b_mu_sim.write(episode, tf.cast(b_mu, dtype))
    #     #self.w_value_sim = self.w_value_sim.write(episode, tf.cast(w_value, dtype))
    #     #self.b_value_sim = self.b_value_sim.write(episode, tf.cast(b_value, dtype))
    #     if self.params['monitor_elig_grads']:
    #         self.w_mu_elig = self.w_mu_elig.write(episode, tf.cast(w_mu_elig_ep, dtype))
    #         self.b_mu_elig = self.b_mu_elig.write(episode, tf.cast(b_mu_elig_ep, dtype))
    #         #self.w_v_elig = self.w_v_elig.write(episode, tf.cast(w_v_elig_ep, dtype))
    #         self.d_dw_mu = self.d_dw_mu.write(episode, tf.cast(d_dw_mu_ep, dtype))
    #         self.d_db_mu = self.d_db_mu.write(episode, tf.cast(d_db_mu_ep, dtype))
    #         #self.d_dw_v = self.d_dw_v.write(episode, tf.cast(d_dw_v_ep, dtype))
    #         #self.d_db_v = self.d_db_v.write(episode, tf.cast(d_db_v_ep, dtype))




class SensoryFeedBack:
    """Creating and memorising sensory feed back information """

    def __init__(self, params):
        super(SensoryFeedBack, self).__init__()
        self.value_net = params.value_net
        self.R1 = tf.Variable(params.R1, trainable=False, dtype=tf.int32)
        self.R2 = tf.Variable(params.R2, trainable=False, dtype=tf.int32)
        self.memory = tf.Variable(tf.zeros((params.batch_size, 1, 3*params.tau_gamma)), trainable=False)
        self.memory_tmp = tf.Variable(tf.zeros((params.batch_size, 1, 3*params.tau_gamma)), trainable=False)



    def update(self, x_t):
        sfb_t = x_t[:, :, self.R2.value()] - x_t[: ,: ,self.R1.value()]
        memo_t = tf.concat([sfb_t[None, :, :], self.memory.value()[:, :, :-1]], axis=2)
        self.memory.assign(memo_t)
        self.memory_tmp.assign(self.memory.value())

    def update_tmp(self, x_t):
        sfb_t = x_t[:, :, self.R2.value()] - x_t[: ,: ,self.R1.value()]
        memo_t = tf.concat([sfb_t[None, :, :], self.memory.value()[:, :, :-1]], axis=2)
        self.memory_tmp.assign(memo_t)

    def value(self):
        if self.value_net=='SFB':
            ret = self.memory.value()
        else:
            ret = None
        return ret






class Gradients:
    """Gradient accumulation handling in training loop"""

    def __init__(self, trainable_variables, params):
        self.train_vars = trainable_variables
        self.grads = [tf.zeros_like(self.train_vars[i]) for i in range(len(self.train_vars))]
        self.value_only = params.train_value_only

    def reset(self):
        self.grads = [tf.zeros_like(self.train_vars[i]) for i in range(len(self.train_vars))]

    def accumulate(self, grads_t):
        if self.value_only:
            for i,grad_t in enumerate(grads_t):
                if i>=2:
                    self.grads[i] += grad_t
        else:
            for i,grad_t in enumerate(grads_t):
                self.grads[i] += grad_t

    def value(self):
        return self.grads
