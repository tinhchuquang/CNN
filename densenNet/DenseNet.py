import numpy as np
import tensorflow as tf
import time
from datetime import timedelta
import DenseNet_load_data
from const import *
import os

def input():
    x = tf.placeholder(tf.float32, [None, width_image, height_image, num_channel])
    y_ = tf.placeholder(tf.float32, [None, num_class])
    is_training = tf.placeholder(tf.bool)

    return x, y_, is_training

def conv2d(x, out_filter, filter_size, strides, padding='SAME'):
    in_filters = (int)(x.get_shape()[-1])
    filter = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filter], tf.float32, tf.contrib.layers.variance_scaling_initializer())
    return tf.nn.conv2d(x, filter, [1, strides, strides, 1], padding)

def avg_pool(x, filter_size, strides):
    return tf.nn.avg_pool(x, [1, filter_size, filter_size, 1], [1, strides, strides, 1], padding='VALID')

def max_pool(x, filter_size, strides):
    return tf.nn.max_pool(x, [1, filter_size, filter_size, 1], [1, strides, strides, 1], padding='SAME')

def dropout(_input, is_training, keep_prob=1):
    if keep_prob < 1:
        output = tf.cond(
            is_training,
            lambda : tf.nn.dropout(_input, keep_prob),
            lambda : _input,
        )
    else:
        output = _input
    return output


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

def composite_function(x, out_filters, is_training, filter_size=3):
    in_filters = (int)(x.get_shape()[-1])
    with tf.variable_scope("composite_function"):
        # Batch norm
        output = batch_norm(x, in_filters, is_training)

        #ReLU
        output = tf.nn.relu(output)

        # Conv2d 3x3
        output = conv2d(output, out_filters, filter_size, strides=1)
        #output = dropout(output, is_training, 0.5)

        return output

def bottleneck(x, is_training):
    in_filters = int(x.get_shape()[-1])
    out_filters = growth_rate *4
    with tf.variable_scope("bottleneck"):
        # Batch normalization
        output = batch_norm(x, in_filters, is_training)

        # ReLu
        output = tf.nn.relu(output)

        # Conv2d 1x1
        output = conv2d(output, out_filters, filter_size=1, strides=1, padding='SAME')
        #output = dropout(output, is_training)

    return output

def add_internal_layer(x, is_training):
    if bc_mode:
        output = bottleneck(x, is_training)
        output = composite_function(output, growth_rate, is_training, filter_size=3)
    else:
        output = composite_function(x, growth_rate, filter_size=3)

    # print(x.get_shape(), ' ', output.get_shape())
    return tf.concat(axis=3, values=(x, output))

def add_block(input, num_layer, is_training):
    output = input
    for layer in range(num_layer):
        with tf.variable_scope("Layer%d" % layer):
            output = add_internal_layer(output, is_training)

            # print("Layer '", str(layer), "' :" ,output.get_shape())
    return output

def transition_layer(x, is_training):
    out_filters = int(int(x.get_shape()[-1])*reduction)
    output = composite_function(x, out_filters, is_training, filter_size=1)
    output = avg_pool(output, 2, 2)

    return output

def last_transition_layer(x, is_training):
    in_filters = int(x.get_shape()[-1])
    # BN
    output = batch_norm(x, in_filters, is_training)

    # Relu
    output = tf.nn.relu(output)

    # Avg pooling
    filter_size = 5#x.get_shape()[-2]
    output = avg_pool(output, filter_size, filter_size)

    # Fully connected
    total_features = int(output.get_shape()[-1])
    output = tf.reshape(output, [-1, total_features])
    W = tf.get_variable('DW', [total_features, num_class], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('bias', [num_class], tf.float32, initializer=tf.constant_initializer(0.0))

    logits = tf.matmul(output, W) + b

    return logits

def inference(x, y_, is_training):
    layer_per_block = int((depth-total_block-1) / total_block)
    # First convolutional layer: filter 3x3x(2 * growth_rate)

    with tf.variable_scope("Init_conv"):
        output = conv2d(x, out_filter=2* growth_rate, filter_size=3, strides=2)

    # Dense block
    for block in range(total_block):
        with tf.variable_scope("Block_%d" % block):
            output = add_block(output, layer_per_block, is_training)

            print("Shape after dense block", str(block), ": ", output.get_shape())
            if block != (total_block-1):
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = transition_layer(output, is_training)

                print("Shape after transition", str(block), ": ", output.get_shape())

    with tf.variable_scope("Transition_to_classes"):
        logits = last_transition_layer(output, is_training)
    print("Shape after last block", logits.get_shape())

    prediction = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
    accurary = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return logits, accurary

def losses(logits, y_):
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
    total_loss = weight_decay*l2_loss + cross_entropy
    return cross_entropy, l2_loss, total_loss

def train_step(total_loss, global_step):
    learning_rate = tf.placeholder(tf.float32)
    train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(total_loss, global_step = global_step)
    return train_step, learning_rate


