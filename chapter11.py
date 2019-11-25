import numpy as np
import os
import tensorflow.compat.v1 as tf
import util.reset_graph as reset_graph

tf.disable_eager_execution()

# 使用Ｈｅ初始化权重　
import tensorflow as tf
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
he_init = tf.variance_scaling_initializer()
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                          kernel_initializer=he_init, name="hidden1")

