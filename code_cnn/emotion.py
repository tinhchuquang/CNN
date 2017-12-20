import tensorflow as tf 
import numpy as np
import random 
import cv2
import scipy

EMOTION_FOLDER = '/home/tinh/CNN/m_CNN/multiltask/database/'

def getEmotionImage():
    print('Load emotion image..................')
    X1 = np.load(EMOTION_FOLDER+'train_emotion.npy')
    X2 = np.load(EMOTION_FOLDER+'public_test_emotion.npy')
    X3 = np.load(EMOTION_FOLDER+'private_test_emotion.npy')
    train_data = []
    public_test_data = []
    private_test_data = []

    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        public_test_data.append(X2[i])
    for i in range(X3.shape[0]):
        private_test_data.append(X3[i])    
    print('Done !')
    print('Number of emotion train data: ', str(len(train_data)))
    print('---------------------------------------------------------------')
    return train_data, public_test_data, private_test_data

def random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])
    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad, mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    return new_batch


def random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def random_flip_updown(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.flipud(batch[i])
    return batch


def random_90degrees_rotation(batch, rotations=[0, 1, 2, 3]):
    for i in range(len(batch)):
        num_rotations = random.choice(rotations)
        batch[i] = np.rot90(batch[i], num_rotations)
    return batch


def random_rotation(batch, max_angle):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            angle = random.uniform(-max_angle, max_angle)
            batch[i] = scipy.ndimage.interpolation.rotate(batch[i], angle, reshape=False)
    return batch


def random_blur(batch, sigma_max=5.0):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            sigma = random.uniform(0., sigma_max)
            batch[i] = scipy.ndimage.filters.gaussian_filter(batch[i], sigma)
    return batch


def augmentation(batch, img_size):
    batch = random_crop(batch, (img_size, img_size), 10)
    #batch = random_blur(batch)
    batch = random_flip_leftright(batch)
    batch = random_rotation(batch, 10)

    return batch


def one_hot(index, num_classes):
	assert index < num_classes and index >= 0
	tmp = np.zeros(num_classes, dtype=np.float32)
	tmp[index] = 1.0
	return tmp


def _input():
	x = tf.placeholder(dtype=tf.float32, shape=[None, 48 , 48, 1], name='input')
	y_ = tf.placeholder(dtype=tf.float32, shape=[None, 7], name='label')
	keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
	return x, y_, keep_prob

def _leaky_relu(x, _anpha):
	output = tf.nn.leaky_relu(x, anpha = _anpha)
	return output

def _conv2d(x, out_filters, kernel_size, stride, padding='SAME'):
	in_filters = x.get_shape()[-1]
	kernel = tf.get_variable(name='kernel', dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.001),
							 shape=[kernel_size, kernel_size, in_filters, out_filters])
	bias = tf.constant(0, dtype=tf.float32, shape=[out_filters], name='bias')
	h = tf.nn.conv2d(input=x, filter=kernel, strides=[1, stride, stride, 1], padding=padding, name='conv')
	output = tf.add(h, bias)
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

def _fc_batch_norm(x, out_dim,_phase_train, activation='linear'):
	assert activation == 'linear' or activation == 'relu'
	W = tf.get_variable('W', [x.get_shape()[1], out_dim],
						initializer=tf.truncated_normal_initializer(stddev=0.001))
	b = tf.get_variable('bias', [out_dim], initializer=tf.constant_initializer(0))
	x = tf.nn.xw_plus_b(x, W, b, name='linear')

	if activation == 'relu':
		x = batch_norm(x.get_shape[1], out_dim, _phase_train)
		x = tf.nn.relu(x, name='relu')
	return x


def _fc(x, out_dim, activation='linear'):
	assert activation == 'linear' or activation == 'relu'
	W = tf.get_variable('W', [x.get_shape()[1], out_dim],
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
		output = _conv2d(x, 64, 3, 1)
		output = batch_norm(output, 64, _phase_train)
		output = _relu(output)

	with tf.variable_scope('Block2_conv2'):
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
		output = _fc(output, 7)

	return output

def _average_pool(x, filter, stride, padding_type):
  return tf.nn.avg_pool(x, [1, filter, filter, 1], [1, stride, stride, 1], padding_type)

def _losses(logits, labels):

	l2_loss = 1e-4 * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
	total_loss = tf.add(l2_loss, cross_entropy, name='loss')
	return total_loss


def _train_op(loss, global_step):
	learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
	train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step)
	return learning_rate, train_step

def from_2D_to_3D(_batch_img):
	for i in range(len(_batch_img)):
		_batch_img[i] = _batch_img[i][:,:,np.newaxis]
	return _batch_img

def load_img(path, size):
	pixel_=[]
	for i in range(size):
		pixel_.append(cv2.imread(path+str(i+1)+'.jpg', 0))
	return pixel_

def run():
	sess = tf.InteractiveSession()
	global_step = tf.contrib.framework.get_or_create_global_step()
	x, y_, keep_prob = _input()
	phase_train = tf.placeholder(tf.bool, name='phase_train')
	logits = inference(x, keep_prob, phase_train)
	loss = _losses(logits, y_)
	learning_rate, train_step = _train_op(loss, global_step)
	prediction = tf.nn.softmax(logits)
	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	writer = tf.summary.FileWriter('./summary_2/')
	writer.add_graph(sess.graph)
	tf.summary.scalar('loss', loss)
	tf.summary.scalar('acc', accuracy)
	for var in tf.trainable_variables():
		tf.summary.histogram(var.name, var)
	merge_summary = tf.summary.merge_all()
	sess.run(tf.global_variables_initializer())

	emotion_train, emotion_public_test, emotion_private_test = getEmotionImage()
	# for i in range(len(pixel_train)):
	# 	pixel_train[i] = np.matrix(pixel_train[i])
	# 	pixel_train[i] = pixel_train[i].reshape(48, 48)
	# 	#pixel_train[i] = pixel_train[i][:,:,np.newaxis]
	# for i in range(len(pixel_test)):
	# 	pixel_test[i] = np.matrix(pixel_test[i])
	# 	pixel_test[i] = pixel_test[i].reshape(48, 48)
	# 	#pixel_test[i] = pixel_test[i][:, :, np.newaxis]

	for epoch in range(100):
		index = np.arange(28709)
		np.random.shuffle(index)
		train_img = []
		train_label = []
		for i in index:
			train_img.append(emotion_train[i][0])
			train_label.append(emotion_train[i][1])


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
			batch_img = augmentation(batch_img, 48)
			batch_label = np.asarray(train_label[top:bot])

			ttl, _, acc, s = sess.run([loss, train_step, accuracy, merge_summary],
									  feed_dict={x: batch_img, y_: batch_label, learning_rate: 1e-4, keep_prob:0.5, phase_train:True})
			writer.add_summary(s, int(global_step.eval()))
			mean_loss.append(ttl)
			mean_acc.append(acc)

		mean_loss = np.mean(mean_loss)
		mean_acc = np.mean(mean_acc)
		print '\nTraining loss: %f' % mean_loss
		print 'Training accuracy: %f' % mean_acc



		index = np.arange(3589)
		np.random.shuffle(index)
		test_img = []
		test_label = []
		for i in index:
			test_img.append(emotion_public_test[i][0])
			test_label.append(emotion_public_test[i][1])
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
			batch_img = augmentation(batch_img, 48)
			batch_label = np.asarray(test_label[top:bot])

			ttl, acc = sess.run([loss, accuracy], feed_dict={x: batch_img, y_: batch_label, keep_prob:1, phase_train:False})
			mean_loss.append(ttl)
			mean_acc.append(acc)

		mean_loss = np.mean(mean_loss)
		mean_acc = np.mean(mean_acc)
		print('\nTesting loss: %f' % mean_loss)
		print('Testing accuracy: %f' % mean_acc)

run()