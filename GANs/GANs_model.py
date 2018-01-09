import numpy as np
import tensorflows as tf
from const import *


'''
    layer for D
'''

def conv(name, x, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
        n = filter_size* filter_size * out_filters
        filter = tf.get_variable('DW',  [filter_size, filter_size, in_filters, out_filters],
                                 tf.float32, tf.random_normal_initializer(stddev=WEIGHT_INIT))
        return tf.nn.conv2d(x, filter, [1, strides, strides, 1])

def relu(x, leakiness=0.0):
    return tf.where(tf.less(leakiness, 0.0), leakiness*x, x, name='leakiness_relu')

def batch_norm_fc(x, phase_train):
    phase_train = tf.convert_to_tensor(phase_train, dtype=tf.bool)
    n_out = int(x.get_shape()[1])
    beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=x.dtype), name='beta', trainable=True, dtype=x.dtype)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=x.dtype), name='gamma', trainable=True, dtype=x.dtype)

    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9, name='ema')

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def batch_norm(x, n_out, phase_train=True, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed



def FC(name, x, out_dim, phase_train, activation='relu'):
    assert (activation=='relu') or (activation=='softmax') or (activation=='linear')
    with tf.variable_scope(name):
        dim = x.get_shape().as_list()
        dim = np.prob(dim[1:])

        x = tf.reshape(x, [-1, dim])
        W = tf.get_variable('DW', [x.get_shape()[-1], out_dim],
                            initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT))
        b = tf.get_variable('bias', [out_dim], initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT))
        x = tf.nn.xw_plus_b(x, W, b)

        if USE_BN:
            x = batch_norm_fc(x, phase_train)
        if activation == 'relu':
            x = relu(x)
        else:
            if activation == 'softmax':
                x = tf.nn.softmax(x)

        return x

def max_pool(x, filter, strides):
    return tf.nn.max_pool(x, [1, filter, filter, 1], [1, strides, strides, 1], 'SAME')


'''
    layer for G
'''




