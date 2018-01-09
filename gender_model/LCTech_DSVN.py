import tensorflow as tf
import os
import time
import gender_model as netmodel
from const import *
import numpy as np


SAVE_FOLDER = './save_model/'


class Gender():
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.global_step = tf.contrib.framework.get_or_create_global_step()

        self.x, self.y_ = netmodel.Input()
        self.y_gender_conv,self.phase_train, self.keep_prob = netmodel.NetModelLayer(self.x)
        self.gender_loss, self.l2_loss, self.total_loss = netmodel._cross_entropy_loss(self.y_gender_conv, self.y_)

        self.train_step, self.learning_rate = netmodel.train_op(self.total_loss, self.global_step)

        self.gender_correct_prediction = tf.equal(tf.argmax(self.y_gender_conv, 1), tf.argmax(self.y_, 1))
        self.gender_true_pred = tf.reduce_sum(tf.cast(self.gender_correct_prediction, dtype=tf.float32))

        self.saver = tf.train.Saver()

        print('Restoring exitsted model')
        self.saver.restore(self.sess, SAVE_FOLDER + 'model.ckpt')
        print('OK')

    def gender_predict(self, batch_size):
        lr = 0.001
        BATCH_SIZE = batch_size

        np.random.shuffle(self.data)
        test_img = []
        test_label = []
        for i in range(len(self.data)):
            test_img.append(self.data[i][0])
            test_label.append(self.data[i][1])
        batch_img = np.asarray(test_img)
        batch_label = np.asarray(test_label)


        predict = self.sess.run([self.y_gender_conv],
                                         feed_dict={self.x: batch_img, self.y_: batch_label, self.phase_train: False, self.keep_prob: 1,
                                                    self.learning_rate: lr})
        predict = np.argmax(predict, axis=1)

        return predict
