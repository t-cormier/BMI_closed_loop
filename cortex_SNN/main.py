import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import io
import seaborn as sns

import models as m


def run_sim(params):

    ###### Run the simulation for n_runs trials ######

    reward_rate_run = []
    z_ctx_run = []
    z_str_run = []
    cn_run = []
    T_run = []
    w_run = []
    b_run = []
    w_v_run = []
    b_v_run = []
    value_run = []

    n_runs = params['n_runs']
    for run in range(n_runs):

        print(f'Run# {run}/{n_runs}                      ')

        ## Initialise the Model

        exp_model = m.ExpModel(params)
        z = exp_model.build_thresh(params['baseline_time'])

        print(tf.reduce_mean(z, axis=0))

        ## Train the model

        reward_rates, z_ctx_sim, z_str_sim, cn, T, w, b, value_sim, w_v, b_v = m.train(exp_model, params['n_seconds'])
        reward_rate_run.append(reward_rates.numpy())
        z_ctx_run.append(z_ctx_sim.numpy())
        z_str_run.append(z_str_sim.numpy())
        cn_run.append(cn)
        T_run.append(T)
        w_run.append(w.numpy())
        b_run.append(b.numpy())
        w_v_run.append(w_v.numpy())
        b_v_run.append(b_v.numpy())
        value_run.append(value_sim.numpy())


    ######## create a dictionary to store the data

    data = { 'reward_rate' : reward_rate_run,
             'z_ctx' : z_ctx_run,
             'z_str' : z_str_run,
             'cn' : cn_run,
             'T': T_run,
             'w' : w_run,
             'b' : b_run,
             'w_v' : w_v_run,
             'b_v' : b_v_run,
             'value' : value_run }

    return data


if __name__ == '__main__':

    ##### Parameters

    params = {
        # simulations params :
        'batch_size' : 30,  #bath size for each run of the simulation
        'n_ctx' : 60,              # number of cortical neurons
        'n_str' : 30,              # number of striatal neurons
        'n_runs' : 5,           # number of runs in the simulation
        'n_seconds' : 600,      # number of seconds simulated per run

        # striatum parameters
        'str_refractory' : 10, #ms    # refractory period of the striatal neurons
        'str_base_activity' : 50, #Hz # used to set the initial bias of the striatal neurons

        # cortex parameters
        'tau_z' : 150,          # spike train filter time constant
        'ctx_noise' : 0.05,     # noise coefficient in the cortex
        'B_con' : 0.4,          # connectivity of thalamo-cortical synapses
        'A_con' : 0.1,          # connectivity in the cortex
        'tau_m' : 20,           # membrane potential of LIF neuron
        'v_thresh' : 1.,        # threshold potential of LIF neuron
        'v_reset' : 0.,         # hard reset voltage of LIF neuron
        'n_refractory': 5, #ms  # refractory period of LIF neuron

        # critic parameters
        'gclip' : 1.0,               # global clip norm to avoid exploding gradients
        'value_act_function' : None, # value neuron activation function
        'value_bias' : True,    # use bias on value neuron
        'critic' : 'LSTM',      # critic model, perceptron or LSTM units

        # training parameters
        'baseline_time' : 10000,#ms  # time length of baseline estimation
        'T_percentile' : 95,      # reward threshold choice based on baseline activity ( percentile )
        'learning_rate' : 5e-4,      # learning rate
        'reward_low_bound' : 1.0,   # reward decay after crossing the threshold to allow next reward (1.0 means the activity has to go below 1.0 * T)
        'tau_gamma' : 150,      # discount factor gamma time constant
        'beta' : 0.}            # entropy regularisation temperature parameter


    data = run_sim(params)
    np.save(f'data_run/dataSNN_{n_str}_{n_ctx}_no_entropy_sig_adam_gclip.npy', data)
