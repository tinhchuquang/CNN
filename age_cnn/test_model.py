import input_data
import tensorflow as tf
import numpy as np
import layer as NetModel
from const import *

NUMBER_SMILE_TEST = 1000
NUMBER_GENDER_TEST = 1118

''' PREPARE DATA '''
train_data, test_data = input_data.getImdbImage()
'''--------------------------------------------------------------------------------------------'''


def one_hot(index, num_classes):
    assert index < num_classes and index >= 0
    tmp = np.zeros(num_classes, dtype=np.float32)
    tmp[index] = 1.0
    return tmp


if __name__ == "__main__":
    sess = tf.InteractiveSession()
    global_step = tf.contrib.framework.get_or_create_global_step()

    x, y_= NetModel.Input()

    y_age_conv, phase_train, keep_prob = NetModel.NetModelLayer(x)

    age_loss, l2_loss, loss = NetModel._cross_entropy_loss(y_age_conv, y_)


    age_correct_prediction = tf.equal(tf.argmax(y_age_conv, 1), tf.argmax(y_, 1))
    age_true_pred = tf.reduce_sum(tf.cast(age_correct_prediction, dtype=tf.float32))


    
    np.random.shuffle(test_data)
    print('Restore model')
    saver = tf.train.Saver()
    # saver.restore(sess, SAVE_FOLDER + 'model.ckpt')
    # saver = tf.train.import_meta_graph(SAVE_FOLDER + '/model.ckpt.meta')
    saver.restore(sess, SAVE_FOLDER + 'model.ckpt')

    print('OK')

    number_batch = len(test_data) // BATCH_SIZE
    age_nb_true_pred = 0

    test_img = []
    test_label = []


    for i in range(len(test_data)):
        test_img.append((test_data[i][0]-128) / 255.0)
        test_label.append(test_data[i][1])

    for batch in range(number_batch):
        top = batch * BATCH_SIZE
        bot = min((batch +1)*BATCH_SIZE, len(test_data))
        batch_img = np.asarray(test_img[top:bot])
        batch_label = np.asarray(test_label[top:bot])

        batch_img = input_data.random_crop(batch_img, (48, 48), 10)

        avg_ttl = []
        avg_rgl = []
        avg_age_loss = []

        ttl, age_loss, l2l = sess.run([loss, age_loss, l2_loss], 
                                                feed_dict={x:batch_img, y_:batch_label,
                                                phase_train:False,
                                                keep_prob:1})
        age_nb_true_pred += sess.run(age_true_pred, feed_dict={x:batch_img, y_:batch_label, 
                                            phase_train:False, 
                                            keep_prob:1})


        avg_ttl.append(ttl)
        avg_age_loss.append(age_loss)
        avg_rgl.append(l2l)

    age_train_accuracy = age_nb_true_pred * 1.0 / 150000
    avg_age_loss = np.average(avg_age_loss)
    avg_rgl = np.average(avg_rgl)
    avg_ttl = np.average(avg_ttl)

    print('Age test accuracy: ' + str(age_train_accuracy * 100))
    print('Total loss: ' + str(avg_ttl) + '. L2-loss: ' + str(avg_rgl))
    print('Age loss: ' + str(avg_age_loss))