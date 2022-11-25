import seaborn
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as kr
from keras import layers

from utils import *

np.set_printoptions(precision=3, suppress=True)

tf.random.set_seed(69)

def create_model(input_shape: int):
    model = kr.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=input_shape),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])
    # for our regression task we'll use mean square error loss fn, gradient descent for the optimiser, and Root Mean Squared Error + Mean Absolute Error for our metrics
    sgd_opt = tf.keras.optimizers.SGD()
    model.compile(loss="mse", optimizer=sgd_opt, metrics=["rmse", "mae"])
    return model



if __name__ == '__main__':
    print(tf.__version__)
    model = create_model((9,))
    print(model.summary())
