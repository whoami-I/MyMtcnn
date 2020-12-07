import tensorflow as tf
from tensorflow import keras
import numpy as np

# input = tf.keras.layers.Input(shape=[12, 12, 3])
# print(input)
# x = tf.keras.layers.Conv2D(10, (3, 3),
#                            strides=1)(input)
# x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
# print(x)
#
# x = tf.keras.layers.Conv2D(2, (3, 3),
#                            strides=1)(x)
# print(x)
# model = tf.keras.models.Model([input], [x])

# a = tf.random.normal([1, 2, 1, 1, 1, 1])
# a = tf.squeeze(a, [2, 4])
# print(a.shape)

# a = tf.random.normal([4, 2])
# s = tf.size(a)
# print(s)

a = tf.random.normal(shape=(10, 4))
print(a)
print(a[:, 3])
