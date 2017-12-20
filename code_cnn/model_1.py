import tensorflow as tf 
import numpy as np
import Image
import pandas as pd 

HOME='/home/tinh/CNN/fer2013/fer2013.csv'
def load_csv(path):
	data = pd.read_csv(path)
	label_train = []
	pixel_train = []
	label_test = []
	pixel_test = []
	label_val= []
	pixel_val = []
	for i in range(len(data)):
		if 'Training' in data['Usage'][i]:
			label_train.append(data['emotion'][i])
			pixel_train.append(data['pixels'][i])
		elif 'PrivateTest' in  data['Usage'][i] :
			label_test.append(data['emotion'][i])
			pixel_test.append(data['pixels'][i])
		else    :
			label_val.append(data['emotion'][i])
			pixel_val.append(data['pixels'][i])
	return label_train, pixel_train, label_test, pixel_test


def load_small_csv(path):
	data = pd.read_csv(path)
	label_train = []
	pixel_train = []
	label_test = []
	pixel_test = []
	label_val= []
	pixel_val = []
	for i in range(len(data)):
		if 'Training' in data['Usage'][i]:
			if len(pixel_train) < 10000:
				label_train.append(one_hot(data['emotion'][i], 7))
				pixel_train.append(data['pixels'][i])
		elif 'PrivateTest' in  data['Usage'][i] :
			label_test.append(one_hot(data['emotion'][i], 7))
			pixel_test.append(data['pixels'][i])
		else    :
			label_val.append(one_hot(data['emotion'][i], 7))
			pixel_val.append(data['pixels'][i])


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
	#output = h + bias
	output = tf.nn.relu(h + bias, name='relu')

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

def batch_norm(x):
	output = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=True, scope='bn')
	return output


def inference(x, _keep_prob):
	with tf.variable_scope('Block1'):
		output = _conv2d(x, 32, 5, 1)
		output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling')

	with tf.variable_scope('Block2'):
		output = _conv2d(output, 32, 4, 1)
		output = _average_pool(output, 3, 2, 'SAME')

	with tf.variable_scope('Block3'):
		output = _conv2d(output, 32, 5, 2)
		output = _average_pool(output, 3, 2, 'SAME')

	with tf.variable_scope('FC'):
		output = _flattten(output)
		output = _fc(output, 3072, 'relu')
		output = _drop_out(output, _keep_prob)

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

def run():
	sess = tf.InteractiveSession()
	global_step = tf.contrib.framework.get_or_create_global_step()
	x, y_, keep_prob = _input()
	logits = inference(x, keep_prob)
	loss = _losses(logits, y_)
	learning_rate, train_step = _train_op(loss, global_step)
	prediction = tf.nn.softmax(logits)
	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	writer = tf.summary.FileWriter('./summary_1/')
	writer.add_graph(sess.graph)
	tf.summary.scalar('loss', loss)
	tf.summary.scalar('acc', accuracy)
	merge_summary = tf.summary.merge_all()
	sess.run(tf.global_variables_initializer())

	label_train,pixel_train,label_test, pixel_test = load_csv(HOME)
	for i in range(len(pixel_train)):
		pixel_train[i] = np.matrix(pixel_train[i])
		pixel_train[i] = pixel_train[i].reshape(48, 48)
		pixel_train[i] = pixel_train[i][:,:,np.newaxis]
	for i in range(len(pixel_test)):
		pixel_test[i] = np.matrix(pixel_test[i])
		pixel_test[i] = pixel_test[i].reshape(48, 48)
		pixel_test[i] = pixel_test[i][:, :, np.newaxis]

	for epoch in range(200):
		index = np.arange(28709)
		np.random.shuffle(index)
		train_img = []
		train_label = []
		for i in index:
			train_img.append(pixel_train[i])
			train_label.append(one_hot(label_train[i], 7))


		# np.random.shuffle(smile_train)
		# train_img = []
		# train_label = []
		# for i in range(len(smile_train)):
		#     train_img.append(smile_train[i][0])
		#     train_label.append(one_hot(smile_train[i][1], 2))
		print 'Epoch %d' % epoch
		mean_loss = []
		mean_acc = []
		batch_size = 256
		num_batch = int(len(train_img) // batch_size)
		for batch in range(num_batch):
			print 'Training on batch .............. %d / %d' % (batch, num_batch)
			top = batch * batch_size
			bot = min((batch + 1) * batch_size, len(train_img))
			batch_img = np.asarray(train_img[top:bot])
			batch_label = np.asarray(train_label[top:bot])

			ttl, _, acc, s = sess.run([loss, train_step, accuracy, merge_summary],
									  feed_dict={x: batch_img, y_: batch_label, learning_rate: 1e-4, keep_prob:0.6})
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
			test_img.append(pixel_test[i])
			test_label.append(one_hot(label_test[i], 7))
		# test_img = []
		# test_label = []
		# for i in range(len(smile_test)):
		#     test_img.append(smile_test[i][0])
		#     test_label.append(one_hot(smile_test[i][1], 2))
		mean_loss = []
		mean_acc = []
		batch_size = 256 
		num_batch = int(len(test_img) // batch_size)
		for batch in range(num_batch):
			top = batch * batch_size
			bot = min((batch + 1) * batch_size, len(test_img))
			batch_img = np.asarray(test_img[top:bot])
			batch_label = np.asarray(test_label[top:bot])

			ttl, acc = sess.run([loss, accuracy], feed_dict={x: batch_img, y_: batch_label, keep_prob:0.6})
			mean_loss.append(ttl)
			mean_acc.append(acc)

		mean_loss = np.mean(mean_loss)
		mean_acc = np.mean(mean_acc)
		print('\nTesting loss: %f' % mean_loss)
		print('Testing accuracy: %f' % mean_acc)

run()