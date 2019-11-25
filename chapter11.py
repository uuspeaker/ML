import numpy as np
import os
import tensorflow.compat.v1 as tf
from util import reset_graph

tf.disable_eager_execution()

# 使用Ｈｅ初始化权重　
import tensorflow as tf
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300

X = tf.compat.v1.placeholder(tf.float32, shape=(None, n_inputs), name="X")
hidden1 = tf.keras.layers.Dense(X, n_hidden1, activation="relu", name="hidden1")

