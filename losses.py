import numpy as np
import tensorflow as tf
tf.random.Generator = None


def mse(X, X_):
    return tf.reduce_mean(tf.reduce_mean(tf.math.squared_difference(X, X_), 1))


def kld_loss(mean, logvar):
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), 1))
    return kl_loss


def cdisttf(data_1, data_2):
    prod = tf.math.reduce_sum(
            (tf.expand_dims(data_1, 1) - tf.expand_dims(data_2, 0)) ** 2, 2
        )
    return (prod + 1e-10) ** (1 / 2)


def dist_loss(data, min_dist, max_dist = 20, cut = False):
    pairwise_dist = cdisttf(data, data)
    dist = pairwise_dist - min_dist
    if cut:
        condition = tf.math.less(dist, 0)
        loss = tf.where(condition, 1, tf.math.exp(-dist))
    else:
        bigdist = max_dist - pairwise_dist
        loss =  tf.math.exp(-dist) + tf.math.exp(-bigdist)
    return loss

