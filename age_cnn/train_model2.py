import tensorflow as tf 
import numpy as np
import input_data
from const import *
import os 

''' PREPARE DATA '''
train_data, test_data = input_data.getImdbAgeImage()
'''--------------------------------------------------------------------------------------------'''


def one_hot(index, num_classes):
    assert index < num_classes and index >= 0
    tmp = np.zeros (num_classes, dtype= np.float32)
    tmp[index] = 1.0
    return tmp

def _input():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 48, 48, 1], name='input')
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 8], name='label')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    return x, y_, keep_prob


def _conv2d(x, out_filters, kernel_size, stride, padding='SAME'):
    in_filters = x.get_shape()[-1]
    kernel = tf.get_variable(name='DW', dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.001),
                             shape=[kernel_size, kernel_size, in_filters, out_filters])
    bias = tf.constant(0, dtype=tf.float32, shape=[out_filters], name='bias')
    h = tf.nn.conv2d(input=x, filter=kernel, strides=[1, stride, stride, 1], padding=padding, name='conv')
    output = tf.add(h, bias)
    return output

def _relu(x):
    output = tf.nn.relu(x)
    return output

def _flatten(x):
    shape = x.get_shape().as_list()
    new_shape = np.prod(shape[1:])
    x = tf.reshape(x, [-1, new_shape], name='flatten')

    return x

def _fc_batch_norm(x, out_dim,_phase_train, activation='linear'):
    assert activation == 'linear' or activation == 'relu'
    W = tf.get_variable('DW', [x.get_shape()[1], out_dim],
                        initializer=tf.truncated_normal_initializer(stddev=0.001))
    b = tf.get_variable('bias', [out_dim], initializer=tf.constant_initializer(0))
    x = tf.nn.xw_plus_b(x, W, b, name='linear')

    if activation == 'relu':
        x = batch_norm(x.get_shape[1], out_dim, _phase_train)
        x = tf.nn.relu(x, name='relu')
    return x


def _fc(x, out_dim, activation='linear'):
    assert activation == 'linear' or activation == 'relu'
    W = tf.get_variable('DW', [x.get_shape()[1], out_dim],
                        initializer=tf.truncated_normal_initializer(stddev=0.001))
    b = tf.get_variable('bias', [out_dim], initializer=tf.constant_initializer(0))
    x = tf.nn.xw_plus_b(x, W, b, name='linear')

    if activation == 'relu':
        x = tf.nn.relu(x, name='relu')
    return x

def _drop_out(x, prob):
    output = tf.nn.dropout(x, keep_prob=prob)
    return output

def batch_norm(x, n_out, phase_train):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train, mean_var_with_update ,lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def batch_norm_fc(x, phase_train):
    phase_train = tf.convert_to_tensor(phase_train, dtype=tf.bool)
    n_out = int(x.get_shape()[1])
    beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=x.dtype),name='beta', trainable=True, dtype=x.dtype)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=x.dtype),name='gamma', trainable=True, dtype=x.dtype)

    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9, name='ema')

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def inference(x, _keep_prob, _phase_train):
    
    with tf.variable_scope('Block1_conv1'):
        output = _conv2d(x, 32, 3, 1)
        output = batch_norm(output, 32, _phase_train)
        output = _relu(output)

    with tf.variable_scope('Block1_conv2'):
        output = _conv2d(output, 32, 3, 1)
        output = batch_norm(output, 32, _phase_train)
        output = _relu(output)
        output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling')

    with tf.variable_scope('Block2_conv1'):
        output = _conv2d(output, 64, 3, 1)
        output = batch_norm(output, 64, _phase_train)
        output = _relu(output)

    with tf.variable_scope('Block2_conv2'):
        output = _conv2d(output, 64, 3, 1)
        output = batch_norm(output, 64, _phase_train)
        output = _relu(output)
        # output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling')

    with tf.variable_scope('Block2_conv3'):
        output = _conv2d(output, 64, 3, 1)
        output = batch_norm(output, 64, _phase_train)
        output = _relu(output)
        output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling')

    with tf.variable_scope('Block3_conv1'):
        output = _conv2d(output, 128, 3, 1)
        output = batch_norm(output, 128, _phase_train)
        output = _relu(output)

    with tf.variable_scope('Block3_conv2'):
        output = _conv2d(output, 128, 3, 1)
        output = batch_norm(output, 128, _phase_train)
        output = _relu(output)
        # output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling')

    with tf.variable_scope('Block3_conv3'):
        output = _conv2d(output, 128, 3, 1)
        output = batch_norm(output, 128, _phase_train)
        output = _relu(output)
        output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling')

    with tf.variable_scope('Block4_conv1'):
        output = _conv2d(output, 256, 3, 1)
        output = batch_norm(output, 256, _phase_train)
        output = _relu(output)

    with tf.variable_scope('Block4_conv2'):
        output = _conv2d(output, 256, 3, 1)
        output = batch_norm(output, 256, _phase_train)
        output = _relu(output)

    with tf.variable_scope('Block4_conv3'):
        output = _conv2d(output, 256, 3, 1)
        output = batch_norm(output, 256, _phase_train)
        output = _relu(output)
        output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling')

    with tf.variable_scope('FC1'):
        output = _flatten(output)
        output = batch_norm_fc(output, _phase_train)
        output = _fc(output, 512, 'relu')

    with tf.variable_scope('FC2'):
        output = batch_norm_fc(output, _phase_train)
        output = _fc(output, 512, 'relu')

    with tf.variable_scope('linear'):
        output = _fc(output, 8)

    return output

def _average_pool(x, filter, stride, padding_type):
  return tf.nn.avg_pool(x, [1, filter, filter, 1], [1, stride, stride, 1], padding_type)

def _losses(logits, labels):

    l2_loss = []
    for var in tf.trainable_variables():
        if var.op.name.find(r'DW') > 0:
            l2_loss.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.name, var)
    l2_loss = 1e-3 * tf.add_n(l2_loss)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    total_loss = tf.add(l2_loss, cross_entropy, name='loss')
    return total_loss, l2_loss


def _train_op(loss, global_step):
    learning_rate = tf.train.exponential_decay(INIT_LR, global_step, DECAY_STEP, DECAY_LR_RATE, staircase=True)
    train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9 , use_nesterov=True).minimize(loss,
                                                                                                                 global_step=global_step)
    tf.add_to_collection('learning_rate', learning_rate)
    return train_step


if __name__ == "__main__":
    sess = tf.InteractiveSession()
    global_step = tf.contrib.framework.get_or_create_global_step()
    x, y_, keep_prob = _input()
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    logits = inference(x, keep_prob, phase_train)
    loss, l2_loss = _losses(logits, y_)
    train_step = _train_op(loss, global_step)
    prediction = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    if not os.path.isfile(SAVE_FOLDER+ 'model.ckpt.index'):
        print('Create new model')
        sess.run(tf.global_variables_initializer())
        print('OK')
    else:
        print('Restoring existed model')
        saver.restore(sess, SAVE_FOLDER + 'model.ckpt')
        print('OK')

    writer = tf.summary.FileWriter('./summary_2/')
    writer.add_graph(sess.graph)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', accuracy)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    merge_summary = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    learning_rate = tf.get_collection('learning_rate')[0]

    for epoch in range(1000):

        np.random.shuffle(train_data)
        train_img = []
        train_label = []
        for i in range(len(train_data)):
            train_img.append((train_data[i][0] -128) /255.0)
            train_label.append(train_data[i][1])


        # np.random.shuffle(smile_train)
        # train_img = []
        # train_label = []
        # for i in range(len(smile_train)):
        #     train_img.append(smile_train[i][0])
        #     train_label.append(one_hot(smile_train[i][1], 2))
        print("Learning rate: %f" % learning_rate.eval())
        print 'Epoch %d' % epoch
        mean_loss = []
        mean_acc = []
        batch_size = 128
        num_batch = int(len(train_img) // batch_size)
        for batch in range(num_batch):
            print 'Training on batch .............. %d / %d' % (batch, num_batch)
            top = batch * batch_size
            bot = min((batch + 1) * batch_size, len(train_img))
            batch_img = np.asarray(train_img[top:bot])
            batch_img = input_data.augmentation(batch_img, 48)
            batch_label = np.asarray(train_label[top:bot])

            ttl,l2l, _, acc, s = sess.run([loss, l2_loss, train_step, accuracy, merge_summary],
                                      feed_dict={x: batch_img, y_: batch_label,keep_prob:0.5, phase_train:True})
            writer.add_summary(s, int(global_step.eval()))
            mean_loss.append(ttl)
            mean_acc.append(acc)

        mean_loss = np.mean(mean_loss)
        mean_acc = np.mean(mean_acc)
        print '\nTraining loss: %f' % mean_loss
        print 'Training accuracy: %f' % mean_acc

        saver.save(sess,  SAVE_FOLDER+'model.ckpt')
        print("Save model sucess")

        test_img = []
        test_label = []
        for i in range(len(test_data)):
            test_img.append((test_data[i][0]-128) / 255.0)
            test_label.append(test_data[i][1])

        mean_loss = []
        mean_acc = []
        batch_size = 128 
        num_batch = int(len(test_img) // batch_size)
        for batch in range(num_batch):
            top = batch * batch_size
            bot = min((batch + 1) * batch_size, len(test_img))
            batch_img = np.asarray(test_img[top:bot])
            batch_img = input_data.augmentation(batch_img, 48)
            batch_label = np.asarray(test_label[top:bot])

            ttl, acc = sess.run([loss, accuracy], feed_dict={x: batch_img, y_: batch_label, keep_prob:1, phase_train:False})
            mean_loss.append(ttl)
            mean_acc.append(acc)

        mean_loss = np.mean(mean_loss)
        mean_acc = np.mean(mean_acc)
        print('\nTesting loss: %f' % mean_loss)
        print('Testing accuracy: %f' % mean_acc)
