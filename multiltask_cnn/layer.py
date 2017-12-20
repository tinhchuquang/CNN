import tensorflow as tf
import numpy as np


WEIGHT_INIT = 0.01
NUM_TASKS = 3
SMILE_SIZE = 96
EMOTION_SIZE = 48
GENDER_SIZE = 48
INIT_LR = 0.01
USE_BN = True
BN_DECAY = 0.99
EPSILON = 0.001
WEIGHT_DECAY = 0.0001
DECAY_STEP = 10000
DECAY_LR_RATE = 0.96

''' Weight initialize '''


def weight_conv_variable(shape, name):
    std = shape[0] * shape[1] * shape[2]
    std = np.sqrt(2. / std)
    initial = tf.truncated_normal(shape, stddev=WEIGHT_INIT, mean=0.0)
    return tf.Variable(initial, name=name)


def weight_fc_variable(shape, name):
    std = shape[0]
    std = np.sqrt(2. / std)
    initial = tf.truncated_normal(shape, stddev=WEIGHT_INIT, mean=0.0)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial, name=name)


''' Define some layers '''

def flatten(x):
    dim = x.get_shape().as_list()
    dim = np.prod(dim[1:])
    dime = dim
    x = tf.reshape(x, [-1, dim])
    return x, dime


def conv2d(x, W, stride, padding_type='SAME'):
    return tf.nn.conv2d(x, W, [1, stride, stride, 1], padding_type)

def max_pool(x, filter, stride, padding_type='SAME'):
    return tf.nn.max_pool(x, [1, filter, filter, 1], [1, stride, stride, 1], padding_type)


def batch_normalization(type, input, is_training, decay, variable_averages):
    shape = np.shape(input)
    if type == 'conv':
        gamma = tf.Variable(tf.constant(1., shape=[shape[3]]))
        beta = tf.Variable(tf.constant(0., shape=[shape[3]]))
        batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2])
        pop_mean = tf.Variable(tf.zeros([shape[3]], dtype=tf.float32), trainable=False)
        pop_var = tf.Variable(tf.ones([shape[3]], dtype=tf.float32), trainable=False)
    elif type == 'fc':
        gamma = tf.Variable(tf.constant(1., shape=[shape[1]]))
        beta = tf.Variable(tf.constant(0., shape=[shape[1]]))
        batch_mean, batch_var = tf.nn.moments(input, [0])
        pop_mean = tf.Variable(tf.zeros([shape[1]], dtype=tf.float32), trainable=False)
        pop_var = tf.Variable(tf.ones([shape[1]], dtype=tf.float32), trainable=False)

    def update_mean_var():
        update_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        update_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([update_mean, update_var]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(is_training, update_mean_var, lambda: (pop_mean, pop_var))
    return tf.nn.batch_normalization(input, mean, var, beta, gamma, EPSILON)

def batch_norm(x, n_out, phase_train, scope='bn'):
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
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
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


''' Define model's input '''

def Input():
    x_smile = tf.placeholder(tf.float32, [None, SMILE_SIZE, SMILE_SIZE, 1])
    x_emotion = tf.placeholder(tf.float32, [None, EMOTION_SIZE, EMOTION_SIZE, 1])
    x_gender = tf.placeholder(tf.float32, [None, GENDER_SIZE, GENDER_SIZE, 3])
    y_smile = tf.placeholder(tf.float32, [None, 2])
    y_emotion = tf.placeholder(tf.float32, [None, 7])
    y_gender = tf.placeholder(tf.float32, [None, 2])

    tf.add_to_collection('x_smile', x_smile)
    tf.add_to_collection('y_smile', y_smile)
    tf.add_to_collection('x_emotion', x_emotion)
    tf.add_to_collection('y_emotion', y_emotion)
    tf.add_to_collection('x_gender', x_gender)
    tf.add_to_collection('y_gender', y_gender)
    return x_smile, y_smile, x_emotion, y_emotion, x_gender, y_gender


''' Define CNN2Head model '''

def inference(x_smile, x_emotion, x_gender):
    # Some parameters for batch normalization
    variable_averages = tf.train.ExponentialMovingAverage(BN_DECAY)
    is_training = tf.placeholder(dtype=tf.bool)

    # Modality net for smile image, from size 96 x 96 x 1 -> 48 x 48 x 16
    # Network: conv 3x3x16, conv 3x3x16, max-pool 2x2 stride = 2
    # Output: h_modality_smile_result

    W_modality_smile_conv1 = weight_conv_variable([3, 3, 1, 16], 'W_modality_smile_conv1')
    b_modality_smile_conv1 = bias_variable([16], 'b_modality_smile_conv1')
    h_modality_smile_conv1 = conv2d(x_smile, W_modality_smile_conv1, 1) + b_modality_smile_conv1
    if USE_BN:
        h_modality_smile_conv1 = batch_norm(h_modality_smile_conv1, 16, is_training)
    h_modality_smile_conv1 = tf.nn.relu(h_modality_smile_conv1)

    W_modality_smile_conv2 = weight_conv_variable([3, 3, 16, 16], 'W_modality_smile_conv2')
    b_modality_smile_conv2 = bias_variable([16], 'b_modality_smile_conv2')
    h_modality_smile_conv2 = conv2d(h_modality_smile_conv1, W_modality_smile_conv2, 1) + b_modality_smile_conv2
    if USE_BN:
        h_modality_smile_conv2 = batch_norm(h_modality_smile_conv2, 16, is_training)
    h_modality_smile_conv2 = tf.nn.relu(h_modality_smile_conv2)

    h_modality_smile_result = max_pool(h_modality_smile_conv2, 2, 2)

    # Modality net for emotion image, from size 48 x 48 x 1 -> 48 x 48 x 16
    # Network: conv 1x1x16
    # Output: h_modality_emotion_result

    W_modality_emotion_conv = weight_conv_variable([1, 1, 1, 16], 'W_modality_emotion_conv')
    b_modality_emotion_conv = bias_variable([16], 'b_modality_emotion_conv')
    h_modality_emotion_conv = conv2d(x_emotion, W_modality_emotion_conv, 1) + b_modality_emotion_conv
    if USE_BN:
        h_modality_emotion_conv = batch_norm(h_modality_emotion_conv, 16, is_training)
    h_modality_emotion_result = tf.nn.relu(h_modality_emotion_conv)

    # Modality net for gender image, from size 48 x 48 x 3 -> 48 x 48 x 16
    # Network: conv 1x1x16
    # Output: h_modality_gender_result

    W_modality_gender_conv = weight_conv_variable([1, 1, 3, 16], 'W_modality_gender_conv')
    b_modality_gender_conv = bias_variable([16], 'b_modality_gender_conv')
    h_modality_gender_conv = conv2d(x_gender, W_modality_gender_conv, 1) + b_modality_gender_conv
    if USE_BN:
        h_modality_gender_conv = batch_norm(h_modality_gender_conv, 16, is_training)
    h_modality_gender_result = tf.nn.relu(h_modality_gender_conv)

    # Combine output from modality nets

    x_image = tf.concat([h_modality_smile_result, h_modality_emotion_result, h_modality_gender_result], 0)

    # print(x_image.get_shape())

    # Shared model: BKNet architecture

    ''' Block 1 '''

    with tf.variable_scope('conv1') as scope:
        W_conv1 = weight_conv_variable([3, 3, 16, 32], 'W')
        b_conv1 = bias_variable([32], 'b')
        h_conv1 = conv2d(x_image, W_conv1, 1) + b_conv1
        if USE_BN:
            h_conv1 = batch_norm(h_conv1, 32, is_training)
        h_conv1 = tf.nn.relu(h_conv1, name=scope.name)

    with tf.variable_scope('conv2') as scope:
        W_conv2 = weight_conv_variable([3, 3, 32, 32], 'W')
        b_conv2 = bias_variable([32], 'b')
        h_conv2 = conv2d(h_conv1, W_conv2, 1) + b_conv2
        if USE_BN:
            h_conv2 = batch_norm(h_conv2, 32, is_training)
        h_conv2 = tf.nn.relu(h_conv2, name=scope.name)

    h_pool1 = max_pool(h_conv2, 2, 2)

    ''' Block 2 '''

    with tf.variable_scope('conv3') as scope:
        W_conv3 = weight_conv_variable([3, 3, 32, 64], 'W')
        b_conv3 = bias_variable([64], 'b')
        h_conv3 = conv2d(h_pool1, W_conv3, 1) + b_conv3
        if USE_BN:
            h_conv3 = batch_norm(h_conv3, 64, is_training)
        h_conv3 = tf.nn.relu(h_conv3, name=scope.name)

    with tf.variable_scope('conv4') as scope:
        W_conv4 = weight_conv_variable([3, 3, 64, 64], 'W')
        b_conv4 = bias_variable([64], 'b')
        h_conv4 = conv2d(h_conv3, W_conv4, 1) + b_conv4
        if USE_BN:
            h_conv4 = batch_norm(h_conv4, 64, is_training)
        h_conv4 = tf.nn.relu(h_conv4, name=scope.name)

    h_pool2 = max_pool(h_conv4, 2, 2)

    ''' Block 3 '''

    with tf.variable_scope('conv5') as scope:
        W_conv5 = weight_conv_variable([3, 3, 64, 128], 'W')
        b_conv5 = bias_variable([128], 'b')
        h_conv5 = conv2d(h_pool2, W_conv5, 1) + b_conv5
        if USE_BN:
            h_conv5 = batch_norm(h_conv5, 128, is_training)
        h_conv5 = tf.nn.relu(h_conv5, name=scope.name)

    with tf.variable_scope('conv6') as scope:
        W_conv6 = weight_conv_variable([3, 3, 128, 128], 'W')
        b_conv6 = bias_variable([128], 'b')
        h_conv6 = conv2d(h_conv5, W_conv6, 1) + b_conv6
        if USE_BN:
            h_conv6 = batch_norm(h_conv6, 128, is_training)
        h_conv6 = tf.nn.relu(h_conv6, name=scope.name)

    h_pool3 = max_pool(h_conv6, 2, 2)

    ''' Block 4 '''

    with tf.variable_scope('conv7') as scope:
        W_conv7 = weight_conv_variable([3, 3, 128, 256], 'W')
        b_conv7 = bias_variable([256], 'b')
        h_conv7 = conv2d(h_pool3, W_conv7, 1) + b_conv7
        if USE_BN:
            h_conv7 = batch_norm(h_conv7, 256, is_training)
        h_conv7 = tf.nn.relu(h_conv7, name=scope.name)

    with tf.variable_scope('conv8') as scope:
        W_conv8 = weight_conv_variable([3, 3, 256, 256], 'W')
        b_conv8 = bias_variable([256], 'b')
        h_conv8 = conv2d(h_conv7, W_conv8, 1) + b_conv8
        if USE_BN:
            h_conv8 = batch_norm(h_conv8, 256, is_training)
        h_conv8 = tf.nn.relu(h_conv8, name=scope.name)

    with tf.variable_scope('conv9') as scope:
        W_conv9 = weight_conv_variable([3, 3, 256, 256], 'W')
        b_conv9 = bias_variable([256], 'b')
        h_conv9 = conv2d(h_conv8, W_conv9, 1) + b_conv9
        if USE_BN:
            h_conv9 = batch_norm(h_conv9, 256, is_training)
        h_conv9 = tf.nn.relu(h_conv9, name=scope.name)

    h_pool4 = max_pool(h_conv9, 2, 2)

    # Flatten block
    h_pool4_flat, size_weight = flatten(h_pool4)
    # print(h_pool4_flat.get_shape())

    # Split block to go to branch

    h_smile, h_emotion, h_gender = tf.split(h_pool4_flat, 3, 0)
    print(h_smile.get_shape())
    print(h_emotion.get_shape())
    print(h_gender.get_shape())

    # Smile branch

    '''Dense Layer 1'''
    with tf.variable_scope('smile_fc1') as scope:
        W_smile_fc1 = weight_fc_variable([size_weight, 256], 'W')
        b_smile_fc1 = bias_variable([256], 'b')
        h_smile_fc1 = tf.matmul(h_smile, W_smile_fc1) + b_smile_fc1
        #if USE_BN:
        #    h_smile_fc1 = batch_norm(h_smile_fc1, 256, is_training)
        h_smile_fc1 = tf.nn.relu(h_smile_fc1, name=scope.name)
        keep_prob_smile_fc1 = tf.placeholder(tf.float32)
        h_smile_fc1_drop = tf.nn.dropout(h_smile_fc1, keep_prob_smile_fc1)

    '''Dense Layer 2'''
    with tf.variable_scope('smile_fc2') as scope:
        W_smile_fc2 = weight_fc_variable([256, 256], 'W')
        b_smile_fc2 = bias_variable([256], 'b')
        h_smile_fc2 = tf.matmul(h_smile_fc1_drop, W_smile_fc2) + b_smile_fc2
        #if USE_BN:
        #    h_smile_fc2 = batch_norm(h_smile_fc2, 256, is_training)
        h_smile_fc2 = tf.nn.relu(h_smile_fc2, name=scope.name)
        keep_prob_smile_fc2 = tf.placeholder(tf.float32)
        h_smile_fc2_drop = tf.nn.dropout(h_smile_fc2, keep_prob_smile_fc2)

    ''' Softmax Layer '''
    with tf.variable_scope('smile_softmax') as scope:
        W_smile_fc3 = weight_fc_variable([256, 2], 'W')
        b_smile_fc3 = bias_variable([2], 'b')
        y_smile_conv = tf.matmul(h_smile_fc2_drop, W_smile_fc3) + b_smile_fc3

    # Emotion branch

    '''Dense Layer 1'''
    with tf.variable_scope('emotion_fc1') as scope:
        W_emotion_fc1 = weight_fc_variable([size_weight, 256], 'W')
        b_emotion_fc1 = bias_variable([256], 'b')
        h_emotion_fc1 = tf.matmul(h_emotion, W_emotion_fc1) + b_emotion_fc1
        #if USE_BN:
        #    h_emotion_fc1 = batch_norm(h_emotion_fc1, 256, is_training)
        h_emotion_fc1 = tf.nn.relu(h_emotion_fc1, name=scope.name)
        keep_prob_emotion_fc1 = tf.placeholder(tf.float32)
        h_emotion_fc1_drop = tf.nn.dropout(h_emotion_fc1, keep_prob_emotion_fc1)

    '''Dense Layer 2'''
    with tf.variable_scope('emotion_fc2') as scope:
        W_emotion_fc2 = weight_fc_variable([256, 256], 'W')
        b_emotion_fc2 = bias_variable([256], 'b')
        h_emotion_fc2 = tf.matmul(h_emotion_fc1_drop, W_emotion_fc2) + b_emotion_fc2
        #if USE_BN:
        #    h_emotion_fc2 = batch_norm(h_emotion_fc2, 256, is_training)
        h_emotion_fc2 = tf.nn.relu(h_emotion_fc2, name=scope.name)
        keep_prob_emotion_fc2 = tf.placeholder(tf.float32)
        h_emotion_fc2_drop = tf.nn.dropout(h_emotion_fc2, keep_prob_emotion_fc2)

    ''' Softmax Layer '''
    with tf.variable_scope('emotion_softmax') as scope:
        W_emotion_fc3 = weight_fc_variable([256, 7], 'W')
        b_emotion_fc3 = bias_variable([7], 'b')
        y_emotion_conv = tf.matmul(h_emotion_fc2_drop, W_emotion_fc3) + b_emotion_fc3

    # Gender branch

    '''Dense Layer 1'''
    with tf.variable_scope('gender_fc1') as scope:
        W_gender_fc1 = weight_fc_variable([size_weight, 256], 'W')
        b_gender_fc1 = bias_variable([256], 'b')
        h_gender_fc1 = tf.matmul(h_gender, W_gender_fc1) + b_gender_fc1
        #if USE_BN:
        #    h_gender_fc1 = batch_norm(h_gender_fc1, 256, is_training)
        h_gender_fc1 = tf.nn.relu(h_gender_fc1, name=scope.name)
        keep_prob_gender_fc1 = tf.placeholder(tf.float32)
        h_gender_fc1_drop = tf.nn.dropout(h_gender_fc1, keep_prob_gender_fc1)

    '''Dense Layer 2'''
    with tf.variable_scope('gender_fc2') as scope:
        W_gender_fc2 = weight_fc_variable([256, 256], 'W')
        b_gender_fc2 = bias_variable([256], 'b')
        h_gender_fc2 = tf.matmul(h_gender_fc1_drop, W_gender_fc2) + b_gender_fc2
        #if USE_BN:
        #    h_gender_fc2 = batch_norm(h_gender_fc2, 256, is_training)
        h_gender_fc2 = tf.nn.relu(h_gender_fc2, name=scope.name)
        keep_prob_gender_fc2 = tf.placeholder(tf.float32)
        h_gender_fc2_drop = tf.nn.dropout(h_gender_fc2, keep_prob_gender_fc2)

    ''' Softmax Layer '''
    with tf.variable_scope('gender_softmax') as scope:
        W_gender_fc3 = weight_fc_variable([256, 2], 'W')
        b_gender_fc3 = bias_variable([2], 'b')
        y_gender_conv = tf.matmul(h_gender_fc2_drop, W_gender_fc3) + b_gender_fc3

    print(y_smile_conv.get_shape())
    print(y_emotion_conv.get_shape())
    print(y_gender_conv.get_shape())

    # Define summary op

    loss_summary_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('loss', loss_summary_placeholder)
    acc_train_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('accuracy', acc_train_placeholder)
    summary_op = tf.summary.merge_all()

    # Define L2-loss
    l2_loss_smile = tf.nn.l2_loss(W_modality_smile_conv1) + tf.nn.l2_loss(W_modality_smile_conv2) + tf.nn.l2_loss(
        W_smile_fc1) + tf.nn.l2_loss(W_smile_fc2) + tf.nn.l2_loss(W_smile_fc3)

    l2_loss_emotion = tf.nn.l2_loss(W_emotion_fc1) + tf.nn.l2_loss(W_emotion_fc2) + tf.nn.l2_loss(
        W_emotion_fc3) + tf.nn.l2_loss(W_modality_emotion_conv)

    l2_loss_gender = tf.nn.l2_loss(W_gender_fc1) + tf.nn.l2_loss(W_gender_fc2) + tf.nn.l2_loss(
        W_gender_fc3) + tf.nn.l2_loss(W_modality_gender_conv)

    l2_loss_shared = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(
        W_conv4) + tf.nn.l2_loss(W_conv5) + tf.nn.l2_loss(W_conv6) + tf.nn.l2_loss(W_conv7) + tf.nn.l2_loss(
        W_conv8) + tf.nn.l2_loss(W_conv9)

    regul_loss = WEIGHT_DECAY * (l2_loss_emotion + l2_loss_shared + l2_loss_smile + l2_loss_gender)

    tf.add_to_collection('keep_prob_smile_fc1', keep_prob_smile_fc1)
    tf.add_to_collection('keep_prob_smile_fc2', keep_prob_smile_fc2)
    tf.add_to_collection('keep_prob_emotion_fc1', keep_prob_emotion_fc1)
    tf.add_to_collection('keep_prob_emotion_fc2', keep_prob_emotion_fc2)
    tf.add_to_collection('keep_prob_gender_fc1', keep_prob_gender_fc1)
    tf.add_to_collection('keep_prob_gender_fc2', keep_prob_gender_fc2)
    tf.add_to_collection('y_smile_conv', y_smile_conv)
    tf.add_to_collection('y_emotion_conv', y_emotion_conv)
    tf.add_to_collection('y_gender_conv', y_gender_conv)
    tf.add_to_collection('regul_loss', regul_loss)
    tf.add_to_collection('loss_summary_placeholder', loss_summary_placeholder)
    tf.add_to_collection('acc_train_placeholder', acc_train_placeholder)
    tf.add_to_collection('summary_op', summary_op)
    tf.add_to_collection('is_training', is_training)

    return y_smile_conv, y_emotion_conv, y_gender_conv


''' Define loss function '''


def loss(y_smile_conv, y_smile, y_emotion_conv, y_emotion, y_gender_conv, y_gender):
    regul_loss = tf.get_collection('regul_loss')[0]
    smile_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_smile, logits=y_smile_conv))
    emotion_cross_entropy = 2 * tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_emotion, logits=y_emotion_conv))
    gender_cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_gender, logits=y_gender_conv))

    total_loss = regul_loss + smile_cross_entropy + emotion_cross_entropy + gender_cross_entropy

    tf.add_to_collection('total_loss', total_loss)
    tf.add_to_collection('smile_loss', smile_cross_entropy)
    tf.add_to_collection('emotion_loss', emotion_cross_entropy)
    tf.add_to_collection('gender_loss', gender_cross_entropy)
    return total_loss

def svm_loss(y_smile_conv, y_smile, y_emotion_conv, y_emotion):
    regul_loss = tf.get_collection('regul_loss')[0]
    hinge_loss = tf.maximum(
        y_emotion_conv - tf.reshape(tf.reduce_sum(y_emotion_conv * y_emotion, reduction_indices=1), (-1, 1)) + 1, 0)
    hinge_loss = tf.reduce_mean(tf.reduce_sum(tf.square(hinge_loss), reduction_indices=1) - 1)
    smile_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_smile, logits=y_smile_conv))
    total_loss = hinge_loss + regul_loss + smile_cross_entropy

    tf.add_to_collection('total_loss', total_loss)
    return total_loss


''' Define train op '''

def train_op(loss, global_step):
    learning_rate = tf.train.exponential_decay(INIT_LR, global_step, DECAY_STEP, DECAY_LR_RATE, staircase=True)
    train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.8, use_nesterov=True).minimize(loss,
                                                                                                                   global_step=global_step)
    # train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    tf.add_to_collection('train_step', train_step)
    tf.add_to_collection('learning_rate', learning_rate)
    return train_step