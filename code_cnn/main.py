import tensorflow as tf 
import numpy as np 
import Image
from data_processing import five_crop, shuffle_data, augmentation_batch
import cv2
import os 

SAVE_FOLDER = '/home/tinh/CNN/m_CNN/GENKI4K_SMALL/'

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def load_data():
    all_image = []
    all_label = np.zeros((4000, 1))
    
    for i in range(4000):
        name_image = SAVE_FOLDER + 'file'
        if i < 9:
            name_image = name_image + '000' + str(i+1) +'.jpg'
            all_image.append(load_image(name_image))
        elif i < 99:
            name_image = name_image + '00' + str(i+1) +'.jpg'
            all_image.append(load_image(name_image))
        elif i < 999:
            name_image = name_image + '0'+ str(i+1) + '.jpg'
            all_image.append(load_image(name_image))
        else:
            name_image = name_image +str(i+1) +'.jpg'
            all_image.append(load_image(name_image))
    labels = open('/home/tinh/CNN/m_CNN/GENKI4K/labels.txt', 'r')
    index = 0
    for line in labels:
        all_label[index] = int(line[0])
        index+= 1
    all_label = np.asarray(all_label, dtype='int32')
    #shuffle data
    index = np.arange(4000)
    np.random.shuffle(index)
    train = index[0:3000]
    test = index[3000:]
    # y_train = all_label[train,:]
    # y_test = all_label[test, :]
    y_train = []
    y_test = []
    X_train = []
    X_test = []
    for i in range(4000):
        if i < 3000:
            X_train.append(all_image[index[i]])
            y_train.append(one_hot(all_label[index[i]], 2))
        else :
            X_test.append(all_image[index[i]])
            y_test.append(one_hot(all_label[index[i]], 2))

    return X_train, y_train, X_test, y_test


def load_data2():
    all_image = []
    all_label = np.zeros((4000, 1))
    
    for i in range(4000):
        name_image = SAVE_FOLDER
        if i < 9:
            name_image = name_image + str(i+1) +'.jpg'
            all_image.append(cv2.imread(name_image, 0))
        elif i < 99:
            name_image = name_image +str(i+1) +'.jpg'
            all_image.append(cv2.imread(name_image, 0))
        elif i < 999:
            name_image = name_image + str(i+1) + '.jpg'
            all_image.append(cv2.imread(name_image, 0))
        else:
            name_image = name_image +str(i+1) +'.jpg'
            all_image.append(cv2.imread(name_image, 0))
    labels = open('/home/tinh/CNN/m_CNN/GENKI4K/labels.txt', 'r')
    index = 0
    for line in labels:
        all_label[index] = int(line[0])
        index+= 1
    all_label = np.asarray(all_label, dtype='int32')
    #shuffle data
    index = np.arange(4000)
    np.random.shuffle(index)
    train = index[0:3000]
    test = index[3000:]
    # y_train = all_label[train,:]
    # y_test = all_label[test, :]
    y_train = []
    y_test = []
    X_train = []
    X_test = []
    for i in range(4000):
        if i < 3000:
            X_train.append(all_image[index[i]])
            y_train.append(all_label[index[i]])
        else :
            X_test.append(all_image[index[i]])
            y_test.append(all_label[index[i]])

    return X_train, y_train, X_test, y_test



def one_hot(index, num_classes):
    assert index < num_classes and index >= 0
    tmp = np.zeros(num_classes, dtype=np.float32)
    tmp[index] = 1.0
    return tmp


def _input():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 96 , 96, 1], name='input')
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='label')
    return x, y_


def _conv2d(x, out_filters, kernel_size, stride, padding='SAME'):
    in_filters = x.get_shape()[-1]
    kernel = tf.get_variable(name='DW', dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.001),
                             shape=[kernel_size, kernel_size, in_filters, out_filters])
    bias = tf.constant(0, dtype=tf.float32, shape=[out_filters], name='bias')
    h = tf.nn.conv2d(input=x, filter=kernel, strides=[1, stride, stride, 1], padding=padding, name='conv')
    output = tf.add(h,  bias)
    #output = tf.nn.relu(h + bias, name='relu')

    return output

def _relu(x):
    output = tf.nn.relu(x)
    return output

def _flattten(x):
    shape = x.get_shape().as_list()
    new_shape = np.prod(shape[1:])
    x = tf.reshape(x, [-1, new_shape], name='flatten')
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

def _drop_out(x):
    output = tf.nn.dropout(x, keep_prob=0.7)
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

def inference(x, _phase_train):
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
        output = _conv2d(x, 64, 3, 1)
        output = batch_norm(output, 64, _phase_train)
        output = _relu(output)

    with tf.variable_scope('Block2_conv2'):
        output = _conv2d(x, 64, 3, 1)
        output = batch_norm(output, 64, _phase_train)
        output = _relu(output)

    with tf.variable_scope('Block2_conv3'):
        output = _conv2d(x, 64, 3, 1)
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
        output = _flattten(output)
        output = batch_norm_fc(output, _phase_train)
        output = _fc(output, 256, 'relu')

    with tf.variable_scope('FC2'):
        output = batch_norm_fc(output, _phase_train)
        output = _fc(output, 256, 'relu')

    with tf.variable_scope('linear'):
        output = _fc(output, 2)

    return output



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
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step)
    return learning_rate, train_step

def from_2D_to_3D(_batch_img):
    for i in range(len(_batch_img)):
        _batch_img[i] = _batch_img[i][:,:,np.newaxis]
    return _batch_img

def run():
    sess = tf.InteractiveSession()
    global_step = tf.contrib.framework.get_or_create_global_step()
    x, y_ = _input()
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    logits = inference(x, phase_train)
    loss, l2_loss = _losses(logits, y_)
    learning_rate, train_step = _train_op(loss, global_step)
    prediction = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    saver = tf.train.Saver()
    if not os.path.isfile( '/smile_model/model.ckpt.index'):
        print('Create new model')
        sess.run(tf.global_variables_initializer())
        print('OK')
    else:
        print('Restoring existed model')
        saver.restore(sess, '/smile_model/model.ckpt')
        print('OK')

    writer = tf.summary.FileWriter('./summary/')
    writer.add_graph(sess.graph)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', accuracy)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    merge_summary = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    train_pixel, label_train, test_pixel, label_test = load_data2()
# for i in range(3000):
#   train_pixel[i] = train_pixel[i][:,:,np.newaxis]
# for i in range(1000):
#   test_pixel[i] = test_pixel[i][:, :, np.newaxis]

    for epoch in range(200):
        index = np.arange(3000)
        np.random.shuffle(index)
        train_img = []
        train_label = []
        for i in index:
            train_img.append(train_pixel[i])
            train_label.append(one_hot(label_train[i], 2))

    # np.random.shuffle(smile_train)
    # train_img = []
    # train_label = []
    # for i in range(len(smile_train)):
    #     train_img.append(smile_train[i][0])
    #     train_label.append(one_hot(smile_train[i][1], 2))
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
            batch_img = augmentation_batch(batch_img, 96)
            batch_img = from_2D_to_3D(batch_img)
            batch_label = np.asarray(train_label[top:bot])

            ttl, _, acc, s = sess.run([loss, train_step, accuracy, merge_summary],
                                      feed_dict={x: batch_img, y_: batch_label, learning_rate: 1e-3, phase_train:True})
            writer.add_summary(s, int(global_step.eval()))
            mean_loss.append(ttl)
            mean_acc.append(acc)

        mean_loss = np.mean(mean_loss)
        mean_acc = np.mean(mean_acc)
        print '\nTraining loss: %f' % mean_loss
        print 'Training accuracy: %f' % mean_acc

        saver.save(sess,  '/smile_model/model.ckpt')
        print("Save model sucess")

        index = np.arange(1000)
        np.random.shuffle(index)
        test_img = []
        test_label = []
        for i in index:
            test_img.append(test_pixel[i])
            test_label.append(one_hot(label_test[i], 2))
        # test_img = []
        # test_label = []
        # for i in range(len(smile_test)):
        #     test_img.append(smile_test[i][0])
        #     test_label.append(one_hot(smile_test[i][1], 2))
        mean_loss = []
        mean_acc = []
        batch_size = 128 
        num_batch = int(len(test_img) // batch_size)
        for batch in range(num_batch):
            top = batch * batch_size
            bot = min((batch + 1) * batch_size, len(test_img))
            batch_img = np.asarray(test_img[top:bot])
            batch_img = augmentation_batch(batch_img, 96)
            batch_img = from_2D_to_3D(batch_img)
            batch_label = np.asarray(test_label[top:bot])

            ttl, acc = sess.run([loss, accuracy], feed_dict={x: batch_img, y_: batch_label, phase_train:False})
            mean_loss.append(ttl)
            mean_acc.append(acc)

        mean_loss = np.mean(mean_loss)
        mean_acc = np.mean(mean_acc)
        print('\nTesting loss: %f' % mean_loss)
        print('Testing accuracy: %f' % mean_acc)

run()