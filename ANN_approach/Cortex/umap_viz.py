import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import umap
import numpy as np
import io



def load_data(filename):
    data = np.load(f'{filename}.npy')
    return data

x_hist = load_data('RNN_100')

# visualize the evolution of the net in low dimension.
mapper  = umap.UMAP(n_neighbors=5).fit_transform(x_hist)
mapper_1 = mapper[:100, :]
mapper_2 = mapper[100:200, :]
mapper_3 = mapper[200:300, :]
mapper_4 = mapper[300:400, :]
mapper_5 = mapper[400:500, :]
plt.plot(*mapper_1.T, alpha=0.5)
plt.plot(*mapper_2.T, alpha=0.5)
plt.plot(*mapper_3.T, alpha=0.5)
plt.plot(*mapper_4.T, alpha=0.5)
plt.plot(*mapper_5.T, alpha=0.5)
plt.show()
