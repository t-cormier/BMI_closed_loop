import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import io



class ValueEstimator(keras.layers.Layer):
    """ custom gradient layer for value estimation and autodiff """

    def __init__(self, params, n_hidden):
        super(ValueEstimator, self).__init__()

        self.params = params
        self.n_hidden = n_hidden
        self.elig_w = tf.Variable(tf.zeros((n_hidden, 1)), trainable=False, name='elig_w_backprop')
        self.w_V_elig = tf.Variable(tf.zeros((n_hidden, 1)), trainable=False, name='elig_w')
        self.td = tf.Variable(0., trainable=False, name='td_memo_value_est')

        w_init = keras.initializers.GlorotUniform(seed=1234)(shape=(n_hidden, 1))
        b_init = tf.zeros((1,))
        self.w = tf.Variable(w_init, name='w_V')
        self.b = tf.Variable(b_init, name='b_V')

    def convolve(self, x_bar, new_x, decay):
        return x_bar * decay + new_x

    def update_elig(self, x):
        self.elig_w.assign(self.convolve(self.elig_w.value(), self.w.value(), self.params.gamma))
        self.w_V_elig.assign(self.convolve(self.w_V_elig.value(), tf.squeeze(x)[:,None], self.params.gamma))

    @tf.custom_gradient
    def call(self, x):



        def grad(dy, variables):
            dL_dx = - self.td.value() * self.elig_w.value()
            dL_dx = tf.reshape(dL_dx, [self.params.batch_size, 1, self.n_hidden])

            dL_dw = tf.squeeze(- tf.multiply(self.td.value(), self.w_V_elig.value()))[:, None]
            dL_dw = tf.reshape(dL_dw, [self.n_hidden, 1])
            dL_db = tf.squeeze(- self.td.value() )[None] #/(1-gamma)
            dL_db = tf.reshape(dL_db, [1,])

            return [dL_dx], [dL_db, dL_dw]

        V = tf.matmul(x, self.w.value()) + self.b


        return V, grad



class ActorCriticNet(keras.Model):
    def __init__(self, params):
        super(ActorCriticNet, self).__init__()
        self.batch_size = params.batch_size
        self.n_str = params.n_str
        self.n_ctx = params.n_ctx
        self.params = params

        # network
        self.x_shape = (self.batch_size, 1, self.n_ctx)

        if params.css_exc :
            kernel_init_policy = tf.keras.initializers.RandomUniform(minval=0., maxval=params.w_init_exc, seed=1234)
        else :
            kernel_init_policy = tf.keras.initializers.Orthogonal(gain=params.w_init, seed=1234)
        bias_init_policy = tf.keras.initializers.RandomUniform(minval=-params.random_bias_gain, maxval=params.random_bias_gain, seed=1234)



        self.policy = layers.Dense(self.n_str,
                                   kernel_initializer=kernel_init_policy,
                                   bias_initializer=bias_init_policy,
                                   activation=None,
                                   use_bias=True,
                                   name='policy')


        if params.value_net == 'FFD':
            self.value_h1 = layers.Dense(self.n_ctx//10, activation = 'relu', name='hv1')
            self.value_h2 = layers.Dense(self.n_ctx//20, activation = 'relu', name='hv2')
            n_hidden = self.n_ctx//20

        if params.value_net == 'SFB':
            self.value_h1 = layers.Dense(3*params.tau_gamma, activation = 'relu', name='hv1_sfb')
            self.value_h2 = layers.Dense(3*params.tau_gamma, activation = 'relu', name='hv2_sfb')
            n_hidden = 3 * params.tau_gamma

        # TODO:
        # if params.value_net == 'LSTM':
        #     self.value_h = layers.LSTM()

        #self.value_estimator = layers.Dense(1, activation = None, name='value')
        self.value_estimator = ValueEstimator(params, n_hidden)

    def build_weights(self):
        zero = tf.zeros(shape=self.x_shape)
        pre_init_policy = self.policy(zero)
        #pre_init_noise = self.noise(zero)
        if self.params.value_net == 'FFD':
            pre_h1 = self.value_h1(zero)
            pre_h2 = self.value_h2(pre_h1)
        if self.params.value_net == 'SFB':
            sfb_zero = tf.zeros(shape=(self.batch_size, 1, 3*self.params.tau_gamma))
            pre_h1 = self.value_h1(sfb_zero)
            pre_h2 = self.value_h2(pre_h1)
        pre_init_value = self.value_estimator(pre_h2)


    def call(self, x, sfb=None, bootstrap=False):
        mu = self.policy(x)
        mu = tf.reshape(mu, [self.batch_size, 1, self.n_str])

        if self.params.value_net == 'FFD':
            v_h1 = self.value_h1(self.params.value_gain * x)
            v_h2 = self.value_h2(v_h1)

        if self.params.value_net == 'SFB':
            v_h1 = self.value_h1(self.params.value_gain * sfb)
            v_h2 = self.value_h2(v_h1)

        if not bootstrap:
            self.value_estimator.update_elig(v_h2)

        v = self.value_estimator(v_h2)

        v = tf.reshape(v, [self.batch_size, 1, 1])

        #sigma = self.noise(x)
        #sigma = tf.reshape(sigma, [self.batch_size, 1, self.n_str])

        return mu, v # (mu, sigma, v) if one wants to learn the noise




class Agent(keras.Model):
    def __init__(self, params):
        super(Agent, self).__init__()
        self.params = params

        # net
        self.actor_critic = ActorCriticNet(params)
        self.actor_critic.compile(keras.optimizers.SGD(learning_rate=params.learning_rate))
        self.actor_critic.build_weights()

        # params used
        self.batch_size = params.batch_size
        self.n_str = params.n_str
        self.gamma = params.gamma

        # store action
        self.action = tf.Variable(tf.zeros((self.batch_size, 1, self.n_str)), trainable=False, name='stored_f')

        # noise variable
        self.sigma_init = params.sigma_init
        self.sigma = tf.Variable(params.sigma_init, trainable=False, name='sigma')
        self.decay_sigma = tf.exp(-1/params.tau_sigma)

        # RL eligibility traces
        self.w_mu_elig = tf.Variable(tf.zeros_like(self.actor_critic.policy.kernel), trainable=False)
        self.b_mu_elig = tf.Variable(tf.zeros_like(self.actor_critic.policy.bias), trainable=False)
        #self.w_sigma_elig = tf.Variable(tf.zeros_like(self.actor_critic.noise.kernel), trainable=False)
        #self.b_sigma_elig = tf.Variable(tf.zeros_like(self.actor_critic.noise.bias), trainable=False)
        #self.w_V_elig = tf.Variable(tf.zeros_like(self.actor_critic.value_estimator.kernel), trainable=False)


    def compute_f(self, mu, sigma):
        f = tf.random.normal(shape=((self.batch_size, 1, self.n_str)), mean=mu, stddev=tf.math.sqrt(2.)*sigma)
        return f

    def update_sigma(self, reward):
        if reward>0.:
            self.sigma.assign(self.sigma.value() * self.decay_sigma)
        else :
            self.sigma.assign(self.sigma.value() * (2 - self.decay_sigma))

        self.sigma.assign(tf.math.minimum( self.sigma_init, self.sigma.value()))

    def call(self, x, actor, sfb, test_lr=False):
        if actor and test_lr==False:
            mu, _ = self.actor_critic(x, sfb=sfb)
            f = self.compute_f(mu, self.sigma.value())
            self.action.assign(f)
        else :
            f = self.action.value()
        return f

    def convolve(self, x_bar, new_x, decay):
        return x_bar * decay + new_x


    def update_RL_elig(self, f, mu, sigma, x, actor):
        x_T = tf.transpose(x, perm=[0, 2, 1])
        actor_float = tf.cast(actor, tf.float32)
        d_dw_mu = tf.squeeze(tf.matmul(x_T, actor_float * (f-mu))) # /(sigma**2)))
        d_db_mu = tf.squeeze( actor_float * (f-mu)) #/(sigma**2))
        #d_dw_sigma = tf.squeeze(tf.matmul(x_T, actor * ( (f-mu)**2  / (sigma**3) - 1/sigma)))
        #d_db_sigma = tf.squeeze( actor * ( (f-mu)**2 / (sigma**3) - 1/sigma))

        #d_dw = [d_dw_mu, d_db_mu] #, d_dw_sigma, d_db_sigma]

        self.w_mu_elig.assign(self.convolve(self.w_mu_elig.value(), d_dw_mu, self.gamma))
        self.b_mu_elig.assign(self.convolve(self.b_mu_elig.value(), d_db_mu, self.gamma))
        #self.w_sigma_elig.assign(self.convolve(self.w_sigma_elig.value(), d_dw_sigma, self.gamma))
        #self.b_sigma_elig.assign(self.convolve(self.b_sigma_elig.value(), d_db_sigma, self.gamma))
        # self.w_V_elig.assign(self.convolve(self.w_V_elig.value(), tf.squeeze(x)[:,None], self.gamma))



    def compute_gradients(self, td):
        w_mu_grad_t = - tf.squeeze(tf.multiply(td, self.w_mu_elig.value()))
        b_mu_grad_t = - tf.squeeze(tf.multiply(td, self.b_mu_elig.value()))
        #w_sigma_grad_t = tf.squeeze(tf.multiply(td, self.w_sigma_elig.value()))
        #b_sigma_grad_t = tf.squeeze(tf.multiply(td, self.b_sigma_elig.value()))
        # w_V_grad_t = tf.squeeze(- tf.multiply(td, self.w_V_elig.value()))[:, None]
        # b_V_grad_t = tf.squeeze(- td )[None] #/(1-gamma)
        #grads = [tf.zeros_like(self.trainable_variables[i]) for i in range(len(self.trainable_variables))]
        # grads[0] += w_mu_grad_t
        # grads[1] += b_mu_grad_t
        return [w_mu_grad_t, b_mu_grad_t]  #[w_mu_grad_t, b_mu_grad_t]#, w_V_grad_t, b_V_grad_t] # [w_mu_grad_t, b_mu_grad_t, w_sigma_grad_t, b_sigma_grad_t, w_V_grad_t, b_V_grad_t] for learning noise


    def update_value_est_td(self, td):
        self.actor_critic.value_estimator.td.assign(tf.squeeze(td))


    def learn(self, x, x_, reward, actor, sfb=None):

        # would require a reward array for bootstraping and the state after the last reward. in order to compute v_t+1

        mu, v = self.actor_critic(x, sfb=sfb)
        _, v_ = self.actor_critic(x_, sfb=sfb, bootstrap=True)

        td = tf.stop_gradient(reward + self.gamma**(self.params.n_step+1) * v_) - v
        self.update_value_est_td(td)
        #v_loss = td**2
        v_loss = v
        grads_v = tf.gradients(v_loss, self.trainable_variables[2:])
        #grads_v = self.replace_none_with_zeros(grads_v)


        # update elig traces
        self.update_RL_elig(self.action.value(), mu, self.sigma.value(), x, actor)

        # compute gradients
        grads_mu = self.compute_gradients(td)

        #grads = [grads_v[i] + grads_mu[i] for i in range(len(self.trainable_variables))]
        grads = [*grads_mu, *grads_v]
        return mu, v, td, grads
