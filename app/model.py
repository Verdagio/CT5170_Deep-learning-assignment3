import seaborn
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as kr

from utils import *

np.set_printoptions(precision=3, suppress=True)

tf.random.set_seed(69)

def get_model():
    model = kr.Sequential([
        kr.layers.Dense()
    ])
    model = ks.Sequenctial

if __name__ == '__main__':
    print(tf.__version__)
