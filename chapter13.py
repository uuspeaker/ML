import numpy as np
import os
import tensorflow.compat.v1 as tf
from util import reset_graph, save_fig
import matplotlib.pyplot as plt

tf.disable_eager_execution()

def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

def plot_color_image(image):
    plt.imshow(image.astype(np.uint8),interpolation="nearest")
    plt.axis("off")


# 卷基层实现
from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")
print("china", china.shape)
flower = load_sample_image("flower.jpg")
print("flower", flower.shape)
image = china[150:220, 130:250]
height, width, channels = image.shape
image_grayscale = image.mean(axis=2).astype(np.float32)
print("image_grayscale", image_grayscale)
images = image_grayscale.reshape(1, height, width, 1)
print("images", images)

fmap = np.zeros(shape=(7, 7, 1, 2), dtype=np.float32)
fmap[:, 3, 0, 0] = 1
fmap[3, :, 0, 1] = 1
print("fmap",fmap)
plot_image(fmap[:, :, 0, 0])
plt.show()
plot_image(fmap[:, :, 0, 1])
plt.show()

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, height, width, 1))
feature_maps = tf.constant(fmap)
convolution = tf.nn.conv2d(X, feature_maps, strides=[1,1,1,1], padding="SAME")

# with tf.Session() as sess:
#     output = convolution.eval(feed_dict={X: images})
#
# plot_image(images[0, :, :, 0])
# save_fig("china_original", tight_layout=False)
# plt.show()

# 使用tf.layers.conv2d()
reset_graph()

X = tf.placeholder(shape=(None, height, width, channels), dtype=tf.float32)
conv = tf.layers.conv2d(X, filters=2, kernel_size=7, strides=[2,2],
                        padding="SAME")
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    output = sess.run(conv, feed_dict={X: dataset})

plt.imshow(output[0, :, :, 1], cmap="gray") # plot 1st image's 2nd feature map
plt.show()
