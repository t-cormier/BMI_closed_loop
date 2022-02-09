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



def soft_decay(x_bar_pre, x_t, decay):
    return decay * x_bar_pre + (1-decay) * x_t


def n_return(params, agent, env, x_pre, x_bar_pre, f_sample, sfb=None, test_lr=False):
    """ n-step return implementation for the ms_loop function"""

    r_n = tf.zeros((params.batch_size, 1))
    f_t_fix = tf.zeros((params.batch_size, 1, params.n_str))
    r_t_fix = tf.zeros((params.batch_size, 1))
    x_t_fix = tf.zeros((params.batch_size, 1, params.n_ctx))
    x_t = tf.zeros((params.batch_size, 1, params.n_ctx))
    x_bar_t_fix = tf.zeros((params.batch_size, 1, params.n_ctx))
    done_t_fix = False
    done = False
    for t in tf.range(params.n_step):
        f_pre = agent(x_bar_pre, actor=f_sample, sfb=sfb, test_lr=test_lr)
        x_t, r_t, done = env.step(f_pre, x_pre, x_bar_pre)

        if t==0:
            f_t_fix = f_pre
            r_t_fix = r_t
            x_t_fix = x_t
            done_t_fix = done
            x_bar_t_fix = soft_decay(x_bar_pre, x_t, params.x_decay)
        r_n += params.gamma**(tf.cast(t, tf.float32)) * r_t
        x_pre = x_t
        x_bar_pre = soft_decay(x_bar_pre, x_t, params.x_decay)
        if done:
            break
    x_n = x_t
    x_bar_n = x_bar_pre
    return x_n, r_n, x_bar_n, done_t_fix, f_t_fix, r_t_fix, x_t_fix, x_bar_t_fix



@tf.function
def ms_loop(params, agent, env, f_sample, x_pre, x_bar_pre, sfb=None, test_lr=False):
    """ graph function simulating one step of learning (analog to ms) """


    # f_pre = agent(x_pre, actor=f_sample, sfb=sfb)
    # x_t, reward_t, done = env.step(f_pre, x_pre)
    #f_t_ = agent(x_t, actor=False, sfb=sfb)
    #x_next, _, _ = env.step(f_t_, x_t)
    x_n, r_n, x_bar_n, done, f_t, r_t, x_t, x_bar_t = n_return(params, agent, env, x_pre, x_bar_pre, f_sample, sfb=sfb, test_lr=test_lr)
    mu_t, v_t, td_t, grads_t = agent.learn(x_bar_pre, x_bar_n, r_n, actor=f_sample, sfb=sfb)

    return mu_t, v_t, td_t, grads_t, x_t, done, f_t, r_t, x_bar_t





def define_T(params, agent, env):
    """ pre-run a few episode to define the threshold of the baseline behaviour of the cortex """

    x_bar_shape = (params.batch_size, 1, params.n_monitored)
    x_bar_set = tf.zeros((0, *x_bar_shape))

    for it in tqdm.tqdm(range(params.l_baseline), unit='ep', dynamic_ncols=True, desc='Baseline'):

        x_bar_ep = tf.TensorArray(dtype = tf.float32, size = params.t_limit, clear_after_read=False)

        done = False
        t = 0
        x_pre = env.cell.zero_state()
        x_bar_pre = x_pre

        sfb = SensoryFeedBack(params)



        while t<params.t_limit:

            f_sample = (t%params.action_latency==0)

            mu_t, v_t, td_t, grads_t, x_t, done, f_t, reward_t, x_bar_t = ms_loop(params, agent, env, f_sample, x_pre, x_bar_pre, sfb=sfb.value())
            x_bar_ep = x_bar_ep.write(t, tf.gather(x_bar_t, params.monitored_neurons, axis=2))


            x_pre = x_t
            x_bar_pre = x_bar_t

            if params.value_net == 'SFB':
                sfb.update(x_bar_t)

            t += 1


        x_bar_ep = x_bar_ep.stack()
        x_bar_set = tf.concat([x_bar_set, x_bar_ep], axis=0)

    diff = tf.squeeze(x_bar_set[:, :, :, 1] - x_bar_set[:, :, :, 0])
    T = 0.96 * tf.squeeze(tf.reduce_max(diff))
    print(f'T = {T}')
    params.T = T
    env.T.assign(T)








#------------------------------------------------EPISODE LOOP FUNCTION-----------------------------------------------------#






def episode_loop(params, agent, env, grads, test_lr=False):
    """ python function simulating one episode of the task """


    ############# INITIALIZATION #####################
    # data episode in tensorArrays
    x_ep = tf.TensorArray(dtype = tf.float32, size=params.t_limit, clear_after_read=False)
    x_bar_ep = tf.TensorArray(dtype = tf.float32, size=params.t_limit, clear_after_read=False)
    f_ep = tf.TensorArray(dtype = tf.float32, size=params.t_limit, clear_after_read=False)
    v_ep = tf.TensorArray(dtype = tf.float32, size=params.t_limit, clear_after_read=False)
    mu_ep = tf.TensorArray(dtype = tf.float32, size=params.t_limit, clear_after_read=False)
    sigma_ep = agent.sigma.value()
    td_ep = tf.TensorArray(dtype = tf.float32, size=params.t_limit, clear_after_read=False)
    reward_ep = tf.TensorArray(dtype = tf.float32, size=params.t_limit, clear_after_read=False)
    if params.monitor_elig_grads:
        w_mu_elig_ep = tf.TensorArray(dtype = tf.float32, size=params.t_limit, clear_after_read=False)
        b_mu_elig_ep = tf.TensorArray(dtype = tf.float32, size=params.t_limit, clear_after_read=False)
        #w_v_elig_ep = tf.TensorArray(dtype = tf.float32, size=params.t_limit, clear_after_read=False)
        d_dw_mu_ep = tf.TensorArray(dtype = tf.float32, size=params.t_limit, clear_after_read=False)
        d_db_mu_ep = tf.TensorArray(dtype = tf.float32, size=params.t_limit, clear_after_read=False)
        #d_dw_v_ep = tf.TensorArray(dtype = tf.float32, size=params.t_limit, clear_after_read=False)
        #d_db_v_ep = tf.TensorArray(dtype = tf.float32, size=params.t_limit, clear_after_read=False)

    done = False
    t = 0
    x_pre = env.cell.zero_state()
    x_bar_pre = x_pre


    sfb = SensoryFeedBack(params)


    x_shape = (params.batch_size, 1, params.n_monitored)
    x_bar_shape = (params.batch_size, 1, params.n_monitored)
    f_shape = (params.batch_size, 1, params.n_str)
    v_shape = (params.batch_size, 1, 1)
    mu_shape = (params.batch_size, 1, params.n_str)
    td_shape = (params.batch_size, 1, 1)
    reward_shape = (params.batch_size, 1)
    if params.monitor_elig_grads:
        w_mu_elig_shape = (params.n_monitored, params.n_str)
        b_mu_elig_shape = (params.n_str,)
        #w_v_elig_shape = (params.n_ctx, 1)
        d_dw_mu_shape = (params.n_monitored, params.n_str)
        d_db_mu_shape = (params.n_str,)
        #d_dw_v_shape = (params.n_ctx, 1)
        #d_db_v_shape = (1,)
    ##################################################


    ######### EPISODE LOOP ##########
    while (not done) and t<params.t_limit:

        f_sample = (t%params.action_latency==0)


        mu_t, v_t, td_t, grads_t, x_t, done, f_t, reward_t, x_bar_t = ms_loop(params, agent, env, f_sample, x_pre, x_bar_pre, sfb=sfb.value(), test_lr=test_lr)



        grads.accumulate(grads_t)

        # update the data arrays
        x_ep = x_ep.write(t, tf.gather(x_t, params.monitored_neurons, axis=2))
        x_bar_ep = x_bar_ep.write(t, tf.gather(x_bar_t, params.monitored_neurons, axis=2))
        f_ep = f_ep.write(t, f_t)
        v_ep = v_ep.write(t, v_t)
        mu_ep = mu_ep.write(t, mu_t)
        td_ep = td_ep.write(t, td_t)
        reward_ep = reward_ep.write(t, reward_t)
        if params.monitor_elig_grads:
            w_mu_elig_ep = w_mu_elig_ep.write(t, tf.gather(agent.w_mu_elig.value(), params.monitored_neurons, axis=0))
            b_mu_elig_ep = b_mu_elig_ep.write(t, agent.b_mu_elig.value())
            #w_v_elig_ep = w_v_elig_ep.write(t, agent.w_V_elig.value())
            d_dw_mu_ep = d_dw_mu_ep.write(t, tf.gather(grads_t[0], params.monitored_neurons, axis=0))
            d_db_mu_ep = d_db_mu_ep.write(t, grads_t[1])
            #d_dw_v_ep = d_dw_v_ep.write(t, grads_t[2])
            #d_db_v_ep = d_db_v_ep.write(t, grads_t[3])

        t += 1
        x_pre = x_t
        x_bar_pre = x_bar_t
        if params.value_net == 'SFB':
            sfb.update(x_bar_t)
    #################################


    #### DATA COMPLETION FOR SHAPE MATCHING ####
    for t_prime in range(t, params.t_limit):
        x_ep = x_ep.write(t_prime, tf.zeros(shape=x_shape))
        x_bar_ep = x_bar_ep.write(t_prime, tf.zeros(shape=x_bar_shape))
        f_ep = f_ep.write(t_prime, tf.zeros(shape=f_shape))
        v_ep = v_ep.write(t_prime, tf.zeros(shape=v_shape))
        mu_ep = mu_ep.write(t_prime, tf.zeros(shape=mu_shape))
        td_ep = td_ep.write(t_prime, tf.zeros(shape=td_shape))
        reward_ep = reward_ep.write(t_prime, tf.zeros(shape=reward_shape))
        if params.monitor_elig_grads:
            w_mu_elig_ep = w_mu_elig_ep.write(t_prime, tf.zeros(shape=w_mu_elig_shape))
            b_mu_elig_ep = b_mu_elig_ep.write(t_prime, tf.zeros(shape=b_mu_elig_shape))
            #w_v_elig_ep = w_v_elig_ep.write(t_prime, tf.zeros(shape=w_v_elig_shape))
            d_dw_mu_ep = d_dw_mu_ep.write(t_prime, tf.zeros(shape=d_dw_mu_shape))
            d_db_mu_ep = d_db_mu_ep.write(t_prime, tf.zeros(shape=d_db_mu_shape))
            #d_dw_v_ep = d_dw_v_ep.write(t_prime, tf.zeros(shape=d_dw_v_shape))
            #d_db_v_ep = d_db_v_ep.write(t_prime, tf.zeros(shape=d_db_v_shape))
    ##############################################


    #### APPLY GRADIENTS ####
    train_vars = agent.trainable_variables
    agent.actor_critic.optimizer.apply_gradients(zip([g/(t+1) for g in grads.value()], train_vars))
    grads.reset()
    if not params.train_value_only:
        agent.update_sigma(reward_t)
    #########################


    #### GATHER DATA FOR RETURN ####
    datas = [x_ep.stack(), x_bar_ep.stack(), f_ep.stack(), v_ep.stack(), mu_ep.stack(), sigma_ep, td_ep.stack(), reward_ep.stack(), t]
    ret = (datas, train_vars)
    if params.monitor_elig_grads:
        eligs = [w_mu_elig_ep.stack(), b_mu_elig_ep.stack()]#, w_v_elig_ep.stack()]
        grad_datas = [d_dw_mu_ep.stack(), d_db_mu_ep.stack()] #, d_dw_v_ep.stack(), d_db_v_ep.stack()]
        ret = (datas, train_vars, eligs, grad_datas)
    #################################

    return ret
