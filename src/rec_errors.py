import tensorflow as tf


def euclidean_norm_squared(X, axis=None):
    return tf.reduce_sum(tf.square(X), axis=axis)


def squared_euclidean_norm_reconstruction_error(input, output):
    return euclidean_norm_squared(input - output, axis=1)


def mean_squared_euclidean_norm_reconstruction_error(x, y):
    return tf.reduce_mean(squared_euclidean_norm_reconstruction_error(tf.layers.flatten(x), tf.layers.flatten(y)))
