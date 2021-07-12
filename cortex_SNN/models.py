import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import io



class Striatum:
    def __init__(self, params):
        self.batch_size = params['batch_size']
        self.n_ref = params['str_refractory']
        self.u_ref = True
        self.ref= tf.Variable(tf.zeros((self.batch_size, 1, params['n_str'])), trainable=False)
        self.gain_grad = tf.Variable(tf.ones((self.batch_size, 1, params['n_str'])), trainable = False) # gain grad : times 0 or 1 if ref or not
        self.w = tf.Variable(tf.zeros((self.batch_size, params['n_ctx'], params['n_str'])))
        bias = tf.math.log(params['str_base_activity']/(1000-params['str_base_activity']))
        self.b = tf.Variable(bias*tf.ones((self.batch_size, 1, params['n_str'])))  # (batch_size, 1, n_str)


    def sigmoid(self, x):
        return 1/(1+tf.exp(-x))

    def __call__(self, x):
        p_str_batch =  self.sigmoid(tf.matmul(x, self.w.value()) + self.b) # (batch_size, 1, n_str)
        rand = tf.random.uniform(shape=p_str_batch.get_shape()) # (batch_size, 1, str)
        u = tf.where(tf.greater(p_str_batch, rand), tf.ones_like(p_str_batch), tf.zeros_like(p_str_batch)) # (batchsize, 1, str)

        if self.u_ref:

            cond_u = tf.greater(self.ref , self.n_ref)
            u = tf.where(cond_u, u, tf.zeros_like(u))

            new_ref = self.ref.value() + (1 - u)
            new_ref = tf.where(tf.equal(u, 1.), tf.zeros_like(new_ref), new_ref)
            self.ref.assign(new_ref)

            self.gain_grad.assign(tf.where(cond_u, tf.ones_like(u), tf.zeros_like(u)))

        return u, p_str_batch





class Cortex:
    def __init__(self, params):
        self.n_str = params['n_str']
        self.n_ctx = params['n_ctx']
        self.batch_size = params['batch_size']
        self.decay = tf.exp(-1/params['tau_m'])
        self.v_thresh = params['v_thresh']
        self.n_refractory = params['n_refractory']
        self.noise_reg = params['ctx_noise']

        self.B = tf.random.normal((self.n_str, self.n_ctx), stddev = 0.5)
        cond_B = tf.random.uniform((self.n_str, self.n_ctx)) < params['B_con']
        self.B = tf.where(cond_B, self.B, tf.zeros_like(self.B))

        self.A = tf.random.normal((self.n_ctx, self.n_ctx), stddev = self.n_str/self.n_ctx)
        cond_A = tf.random.uniform((self.n_ctx, self.n_ctx)) < params['A_con']
        self.A = tf.where(cond_A , self.A, tf.zeros_like(self.A))
        cond_A_rec = tf.cast(tf.linalg.diag(tf.ones(self.n_ctx)), tf.bool) # remove connections from any neuron to itself
        self.A = tf.where(cond_A_rec, tf.zeros_like(self.A), self.A)

        self.B_batch = tf.stack([ self.B for i in range(self.batch_size)], axis=0) # (batch_size, n_str, n_ctx)
        self.A_batch = tf.stack([ self.A for i in range(self.batch_size)], axis=0) # (batch_size, n_ctx, n_ctx)




    def zero_state(self, batch_size):
        v0 = tf.zeros((batch_size, 1, self.n_ctx))
        r0 = tf.zeros((batch_size, 1, self.n_ctx))
        z0 = tf.zeros((batch_size, 1, self.n_ctx))
        return v0, r0, z0


    def __call__(self, u, state): # LIF model hard reset

        # call the old state of the recurrent
        old_v = state[0]
        old_r = state[1]
        old_z = state[2]

        noise = tf.random.normal((self.batch_size, 1, self.n_ctx))

        # compute the input spikes + the recurrent spikes
        new_v = self.decay * old_v + tf.matmul(old_z, self.A_batch) + tf.matmul(u, self.B_batch) + self.noise_reg * noise
        new_v = tf.where(tf.less(old_r, self.n_refractory), tf.zeros_like(new_v), new_v)

        # make the membrane potential sipke
        new_z = tf.greater(new_v, self.v_thresh)
        new_r = old_r + 1
        new_r = tf.where(new_z, tf.zeros_like(new_r), new_r)
        new_z = tf.cast(new_z, tf.float32)



        return new_v, new_r, new_z



class Critic(keras.Model):
    def __init__(self, params):
        super(Critic, self).__init__()
        self.batch_size = params['batch_size']
        self.opt = keras.optimizers.SGD(lr=params['learning_rate'], global_clipnorm=params['gclip'])
        self.inputs = keras.Input(shape=(self.batch_size, 1, params['n_ctx']))

        self.fc1 = layers.Dense(1,
                                activation=params['value_act_function'],
                                kernel_initializer='zeros',
                                use_bias=True)

    def loss_critic(self, td):
        return td ** 2

    def call(self, state):
        value = self.fc1(state)
        value = tf.reshape(value, [self.batch_size, 1])
        return value



class Critic_LSTM(keras.Model):
    def __init__(self, params):
        super(Critic_LSTM, self).__init__()
        self.batch_size = params['batch_size']
        self.opt = keras.optimizers.Adam(lr= params['learning_rate'], global_clipnorm=params['gclip'])
        self.inputs = keras.Input(shape=(self.batch_size, 1, params['n_ctx']))
        self.lstm = layers.LSTM(60)
        self.prediction = layers.Dense(1, activation=params['value_act_function'])


    def loss_critic(self, td):
        return td ** 2

    def call(self, state):
        memo = self.lstm(state)
        value = self.prediction(memo)
        value = tf.reshape(value, [self.batch_size, 1])
        return value









class ExpModel:
    def __init__(self, params):
        super(ExpModel, self).__init__()
        self.params=params
        self.n_str = params['n_str']
        self.n_ctx = params['n_ctx']
        self.batch_size = params['batch_size']
        self.lr = params['learning_rate']
        self.gamma = tf.exp(-1/params['tau_gamma'])
        self.low_bound = params['reward_low_bound']
        self.beta = params['beta']

        self.striatum = Striatum(params)
        self.cortex = Cortex(params)

        #self.critic = Critic(params)       # choice between LSTM or perceptron for the value estimator
        self.critic = Critic_LSTM(params)
        dummy = self.critic(tf.zeros((self.batch_size, 1, self.n_ctx)))  # dummy initialisation of the critic (forced by graph errors)

        self.cn = tf.Variable(0, trainable=False, name='CN')
        self.T = tf.Variable(0., trainable=False, name='threshold')
        self.T_percentile = params['T_percentile']

        # z_memory
        self.z_decay = tf.exp(-1/params['tau_z'])
        self.z_bar = tf.Variable(tf.zeros((self.batch_size, 1, self.n_ctx)), trainable=False)
        self.reward_refractory = tf.Variable(tf.ones((self.batch_size, 1)), trainable=False) # inverted boolean (1 = False, 0 = True ) for easier computation


    def reward_model(self, z_ctx):
        self.z_bar.assign(self.z_decay * self.z_bar.value() + z_ctx)
        z_condition = self.z_bar.value()[:, :, self.cn.value()]

        reward = tf.where(tf.greater_equal(z_condition, self.T), tf.ones_like(z_condition), tf.zeros_like(z_condition))
        rew_cond = tf.cast(tf.multiply(self.reward_refractory.value(), reward), tf.bool)
        reward = tf.where(rew_cond, tf.ones_like(reward), tf.zeros_like(reward))
        new_reward_ref = tf.where(tf.cast(reward, tf.bool), tf.zeros_like(self.reward_refractory.value()), self.reward_refractory.value())

        ref_update_cond = tf.cast(tf.multiply(tf.cast(tf.less(z_condition, self.T * self.low_bound), tf.float32) , (1 - self.reward_refractory.value())) , tf.bool)
        new_reward_ref = tf.where(ref_update_cond, tf.ones_like(self.reward_refractory.value()), new_reward_ref)
        self.reward_refractory.assign(new_reward_ref)
        return reward


    def build_thresh(self, time):
        z_bar_build_memory = tf.zeros((0, self.batch_size, 1, self.n_ctx))
        z_bar_build = tf.zeros((self.batch_size, 1, self.n_ctx))
        state = self.cortex.zero_state(self.batch_size)

        z = tf.zeros((0, self.batch_size, 1, self.n_ctx))

        for t in range(time):
            print(f'{t}/{time}   ', end='\r')
            u, p_str = self.striatum(z_bar_build)
            state = self.cortex(u, state)
            z = tf.concat([z, state[2][None, :, :, :]], axis=0)
            z_bar_build = self.z_decay * z_bar_build + state[2]
            z_bar_build_memory = tf.concat([z_bar_build_memory, z_bar_build[None, :, :, :]], axis=0)


        z_bar_avg = tf.reduce_mean(z_bar_build_memory, axis = (0,1,2))

        # sort to find cn
        idx_sort = tf.argsort(z_bar_avg)
        cn_choice = idx_sort[(self.n_ctx//4) * 3]
        self.cn.assign(cn_choice)


        # test chance reward hits per second
        z_condition = z_bar_build_memory[time//10:,:,:,self.cn.value()]
        self.T.assign(tfp.stats.percentile(z_condition, q=self.T_percentile))

        state = self.cortex.zero_state(self.batch_size)
        tot_reward = 0
        for t in range(time):
            u, p_str = self.striatum(self.z_bar.value())   # (batch_size, 1, n_str)
            state = self.cortex(u, state)
            reward = self.reward_model(state[2])
            tot_reward += tf.squeeze(tf.reduce_mean(reward, axis=0))

        print(f'Chance reward per second : {tot_reward/10}')

        return z


    def __call__(self, state):
        u, p_str = self.striatum(self.z_bar.value())
        state = self.cortex(u, state)
        reward = self.reward_model(state[2])
        return u, p_str, state, reward






@tf.function
def time_step(exp_model, state, value, conv_a_p_b, conv_a_p_w):

    old_state = state
    old_z_bar = exp_model.z_bar.value()
    u, p_str, state, reward = exp_model(state)


    critic_in = exp_model.z_bar.value()
    new_value = exp_model.critic(critic_in)
    td = reward + exp_model.gamma * new_value - value
    loss_critic = exp_model.critic.loss_critic(td)

    vars_critic = exp_model.critic.trainable_variables
    grad_critic = tf.gradients(loss_critic, vars_critic)




    # Policy gradient
    T_old = tf.transpose(old_z_bar, perm=[0, 2, 1])

    conv_a_p_w = conv_a_p_w * exp_model.gamma + tf.matmul(T_old, exp_model.striatum.gain_grad.value() * (u - p_str)) # (batch_size, n_ctx, n_str)
    delta_w_policy_batch = tf.reduce_mean( exp_model.lr * td[:,:,None] * conv_a_p_w  , axis=0 ) # (n_ctx, str)
    delta_w_policy = tf.stack([delta_w_policy_batch for i in range(exp_model.batch_size)], axis=0) # (batch_size, n_ctx, n_str)

    conv_a_p_b = conv_a_p_b * exp_model.gamma + exp_model.striatum.gain_grad.value() * (u-p_str)
    delta_b_batch = tf.reduce_mean( exp_model.lr * td[:,:,None] * conv_a_p_b  , axis=0 ) # (1, str)
    delta_b = tf.stack([delta_b_batch for i in range(exp_model.batch_size)], axis=0) # (batch_size, 1, n_str)

    # Entropy regularisation
    term = p_str * (1 - p_str) * tf.math.log(p_str / (1 - p_str)) # (batch_size, 1, str)

    delta_w_entropy_batch = tf.reduce_mean(exp_model.lr * tf.matmul(T_old, term), axis=0) # (n_ctx, n_str)
    delta_w_entropy = tf.stack([delta_w_entropy_batch for i in range(exp_model.batch_size)], axis=0) # (batch_size, n_ctx, n_str)

    delta_b_entropy_batch = tf.reduce_mean(exp_model.lr * term, axis = 0)
    delta_b_entropy = tf.stack([delta_b_entropy_batch for i in range(exp_model.batch_size)], axis=0) # (batch_size, 1, n_str)


    delta_w = delta_w_policy - exp_model.beta * delta_w_entropy
    delta_b = delta_b - exp_model.beta * delta_b_entropy


    return  (reward,
             state,
             new_value,
             conv_a_p_b,
             conv_a_p_w,
             delta_w,
             delta_b,
             grad_critic,
             u)



def train(exp_model, n_seconds):

    reward_hits = []
    w_second = []
    b_second = []
    w_v_second = []
    b_v_second = []


    conv_a_p_w = tf.zeros((exp_model.batch_size, exp_model.n_ctx, exp_model.n_str))
    conv_a_p_b = tf.zeros((exp_model.batch_size, 1, exp_model.n_str))
    value = tf.zeros((exp_model.batch_size, 1))

    z_ctx_sim = tf.zeros((0, 1, exp_model.n_ctx)) # (time, )
    z_str_sim = tf.zeros((0, 1, exp_model.n_str))
    value_sim = tf.zeros((0, 1))

    state = exp_model.cortex.zero_state(exp_model.batch_size)

    for second in range(n_seconds):

        tot_reward = 0

        delta_w = tf.zeros_like(exp_model.striatum.w.value())
        delta_b = tf.zeros_like(exp_model.striatum.b.value())
        grad_critic = [tf.zeros_like(exp_model.critic.trainable_variables[i]) for i in range(len(exp_model.critic.trainable_variables))]

        for t in range(1000):

            reward, state, value, conv_a_p_b, conv_a_p_w, delta_w_t, delta_b_t, grad_critic_t, u = time_step(exp_model,
                                                                                                           state,
                                                                                                           value,
                                                                                                           conv_a_p_b,
                                                                                                           conv_a_p_w)

            # data saving
            tot_reward += tf.squeeze(tf.reduce_mean(reward, axis=0))
            z_ctx_sim = tf.concat([z_ctx_sim, state[2][None, 0, :, :]], axis=0)
            z_str_sim = tf.concat([z_str_sim, u[None, 0,:,:]], axis=0)
            value_sim = tf.concat([value_sim, value[None, 0,:]], axis = 0)

            # training update saving
            delta_w += delta_w_t
            delta_b += delta_b_t
            for i,grad in enumerate(grad_critic_t):
                grad_critic[i] += grad




        z_avg = tf.reduce_mean(z_ctx_sim[-1000:-1], axis=0)



        # weight update
        exp_model.striatum.w.assign_add(delta_w)
        exp_model.striatum.b.assign_add(delta_b)
        vars_critic = exp_model.critic.trainable_variables
        exp_model.critic.opt.apply_gradients(zip(grad_critic, vars_critic))

        # Data saving
        w_second.append(exp_model.striatum.w.value())
        b_second.append(exp_model.striatum.b.value())
        w_v_second.append(exp_model.critic.get_weights()[0])
        reward_hits.append(tot_reward)

        print(f'Second#{second} Reward hits:{reward_hits[-1]}                ', end="\r")

    return (tf.stack(reward_hits),
           tf.stack(z_ctx_sim),
           tf.stack(z_str_sim),
           exp_model.cn.value(),
           exp_model.T.value(),
           tf.stack(w_second),
           tf.stack(b_second),
           tf.stack(value_sim),
           tf.stack(w_v_second),
           tf.stack(b_v_second))
