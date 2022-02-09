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
import argparse


import class_agent as ca
import class_env as ce
import class_utils as cu
import fn_train as ft




if __name__ == '__main__':

    # arg parser for eager execution
    parser = argparse.ArgumentParser()
    parser.add_argument('--eager', action='store_true', default=False)
    _args = parser.parse_args()
    if _args:
        tf.config.run_functions_eagerly(True)

    # initialization
    p = cu.Params()
    print(f'sim_ID : {p.sim_ID}')
    data = cu.DataGPU(p)

    env = ce.Env(p)
    env.build()
    agent = ca.Agent(p)
    test_param = p.test_learning

    ft.define_T(p, agent, env)

    # training loop
    for episode in tqdm.tqdm(range(p.n_episode), unit='ep', dynamic_ncols=True, desc='Sim'):

        test_lr = False
        if test_param and episode>950:
            test_lr=True

        grads = cu.Gradients(agent.trainable_variables, p)

        if p.monitor_elig_grads:
            datas, vars, eligs, grad_datas = ft.episode_loop(p, agent, env, grads)
            data_vars = [vars[0], vars[1]]
            data.update(*datas, *data_vars, *eligs, *grad_datas, episode=episode)
        else :
            datas, vars = ft.episode_loop(p, agent, env, grads, test_lr=test_lr)
            data_vars = [vars[0], vars[1]]
            data.update(*datas, *data_vars, episode=episode)



    # save
    data.save(f'data_attempt_{p.sim_ID}')
