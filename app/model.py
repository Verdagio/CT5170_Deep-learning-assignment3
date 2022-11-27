import seaborn
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as kr
import tensorflow_decision_forests as tfdf

from utils import *

np.set_printoptions(precision=3, suppress=True)

tf.random.set_seed(69)


def create_model(input_shape: tuple):
    model = kr.Sequential([
        kr.layers.Dense(64, activation='relu', input_shape=input_shape),
        kr.layers.Dense(64, activation='relu')
        kr.layers.Dense(1)
    ])
    # for our regression task we'll use mean square error loss fn, gradient descent for the optimiser, and Root Mean Squared Error + Mean Absolute Error for our metrics
    sgd_opt = tf.keras.optimizers.SGD()
    model.compile(loss='mean_absolute_error', optimizer=sgd_opt, metrics=["mean_squared_error", "mean_absolute_error"])
    return model

def create_random_forest_model():
    model = tfdf.Keras.RandomForestModel(num_trees=1000)
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), metrics=["mean_squared_error", "mean_absolute_error"])
    return model

def generate_md5s(batches):
    EPOCHS = 1000
    # for every batch in the batches generate an md5
    # additionally - generate an md5 for all months (inclusive)
    file_map = dict()
    years = batches.keys()
    for yr in years:
        months = batches[yr].keys()
        files = dict()
        for mo in months:
            batch = batches[yr].get(mo)
            X, Y = split_attrs_labels(batch)   
      
            model = create_model(X[0].shape)
            
            model.fit(X, Y, epochs=EPOCHS, verbose=0)
            filePath = f"./models/{yr}-{mo}.h5"
            model.save(filePath)
            files[mo] = filePath
        file_map[yr] = files
    
    return file_map


if __name__ == '__main__':
    print(tf.__version__)
    model = create_model((9,))
    print(model.summary())
    rf = create_random_forest_model()
    print(rf.summary)
