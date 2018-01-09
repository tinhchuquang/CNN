import tensorflow as tf
import input_data
import os
import numpy as np
import gender_model as netmodel
from const import *


def one_hot(index, num_classess):
    assert index < num_classess and index >= 0
    tmp = np.zeros(num_classess, dtype=np.float32)
    tmp[index] = 1.0
    return tmp

if __name__ == "__main__":

    sess = tf.InteractiveSession()
    global_step = tf.contrib.framework.get_or_create_global_step()

    x, y_ = netmodel.Input()
    y_gender_conv, phase_train, keep_prob = netmodel.NetModelLayer(x)
    gender_loss, l2_loss, total_loss = netmodel._cross_entropy_loss(y_gender_conv, y_)

    train_step, learning_rate = netmodel.train_op(total_loss, global_step)

    y_gender_conv = tf.nn.softmax(y_gender_conv)
    gender_correct_prediction = tf.equal(tf.argmax(y_gender_conv, 1), tf.argmax(y_, 1))
    gender_true_pred = tf.reduce_sum(tf.cast(gender_correct_prediction, dtype=tf.float32))

    saver = tf.train.Saver()

    print('Restoring exitsted model')
    saver.restore(sess, SAVE_FOLDER + 'model.ckpt')
    print('OK')

    point = 0
    number_batch = NUMBER_TEST_DATA // BATCH_SIZE
    sum_loss = []
    sum_l2 = []
    sum_gl = []
    sum_acc = []

    lr = 0.0001
    for batch in range(number_batch):
        # top = batch * BATCH_SIZE
        # bot = min((batch + 1) * BATCH_SIZE, NUMBER_TEST_DATA)
        data_test, point = input_data.get_test_image(point, BATCH_SIZE)
        np.random.shuffle(data_test)
        test_img = []
        test_label = []
        for i in range(len(data_test)):
            test_img.append(data_test[i][0])
            test_label.append(data_test[i][1])
        batch_img = np.asarray(test_img)
        batch_label = np.asarray(test_label)

        # batch_img = np.asarray(input_data.random_crop(batch_img, (112, 90), 10))
        ttl, gl, l2l, acc = sess.run([total_loss, gender_loss, l2_loss, gender_true_pred],
                                     feed_dict={x: batch_img, y_: batch_label, phase_train: False, keep_prob: 1,
                                                learning_rate: lr})
        sum_loss.append(ttl)
        sum_gl.append(gl)
        sum_l2.append(l2l)
        sum_acc.append(acc)

    sum_loss = np.average(sum_loss)
    sum_gl = np.average(sum_gl)
    sum_l2 = np.average(sum_l2)
    sum_acc = np.average(sum_acc)

    print('Summary test')
    print('Total loss: ' + str(sum_loss) + ', Gender loss: ' + str(sum_gl) + ', L2 loss: ' + str(sum_l2))
    print('Gender test accuracy: ' + str(sum_acc))