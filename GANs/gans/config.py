import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, LeakyReLU,BatchNormalization,Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist
from random import randint
import matplotlib.pyplot as plt
from pathlib import Path
import os

base = (Path(os.path.abspath(__file__)).parent).parent
data_path = base/'data'
models_path = data_path/'models'

if not data_path.exists():
    data_path.mkdir()

if not models_path.exists():
    models_path.mkdir() 