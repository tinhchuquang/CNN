import input_data
import tensorflow as tf
import numpy as np
import NetModel3 as NetModel
from const import *

NUMBER_SMILE_TEST = 1000
NUMBER_GENDER_TEST = 1118

''' PREPARE DATA '''
smile_train, smile_test = input_data.getSmileImage()
emotion_train, emotion_public_test, emotion_private_test = input_data.getEmotionImage()
gender_train, gender_test = input_data.getImdbImage()
'''--------------------------------------------------------------------------------------------'''


def one_hot(index, num_classes):
    assert index < num_classes and index >= 0
    tmp = np.zeros(num_classes, dtype=np.float32)
    tmp[index] = 1.0
    return tmp


if __name__ == "__main__":
    sess = tf.InteractiveSession()
    global_step = tf.contrib.framework.get_or_create_global_step()

    x, y_, mask = NetModel.Input()

    y_smile_conv, y_emotion_conv, y_gender_conv, phase_test, keep_prob = NetModel.NetModelLayer(x)

    smile_loss, emotion_loss, gender_loss, l2_loss, loss = NetModel._cross_entropy_loss(y_smile_conv, y_emotion_conv,
                                                                                     y_gender_conv, y_, mask)

    smile_mask = tf.get_collection('smile_mask')[0]
    emotion_mask = tf.get_collection('emotion_mask')[0]
    gender_mask = tf.get_collection('gender_mask')[0]
    y_smile = tf.get_collection('y_smile')[0]
    y_emotion = tf.get_collection('y_emotion')[0]
    y_gender = tf.get_collection('y_gender')[0]

    smile_correct_prediction = tf.equal(tf.argmax(y_smile_conv, 1), tf.argmax(y_smile, 1))
    emotion_correct_prediction = tf.equal(tf.argmax(y_emotion_conv, 1), tf.argmax(y_emotion, 1))
    gender_correct_prediction = tf.equal(tf.argmax(y_gender_conv, 1), tf.argmax(y_gender, 1))

    smile_true_pred = tf.reduce_sum(tf.cast(smile_correct_prediction, dtype=tf.float32) * smile_mask)
    emotion_true_pred = tf.reduce_sum(tf.cast(emotion_correct_prediction, dtype=tf.float32) * emotion_mask)
    gender_true_pred = tf.reduce_sum(tf.cast(gender_correct_prediction, dtype=tf.float32) * gender_mask)

    test_data = []

    # Mask: Smile -> 0, Emotion -> 1, Gender -> 2
    for i in range(len(smile_test)):
        img = (smile_test[i % 3000][0] - 128) / 255.0
        label = smile_test[i % 3000][1]
        test_data.append((img, one_hot(label, 7), 0.0))
    for i in range(len(emotion_private_test)):
        test_data.append((emotion_public_test[i][0], emotion_public_test[i][1], 1.0))
    for i in range(len(gender_test)):
        img = (gender_test[i][0] - 128) / 255.0
        label = (int)(gender_test[i][1])
        test_data.append((img, one_hot(label, 7), 2.0))
    np.random.shuffle(test_data)
    print('Restore model')
    saver = tf.train.Saver()
    # saver.restore(sess, SAVE_FOLDER + 'model.ckpt')
    # saver = tf.train.import_meta_graph(SAVE_FOLDER + '/model.ckpt.meta')
    saver.restore(sess, SAVE_FOLDER2 + 'model.ckpt')

    print('OK')

    test_img = []
    test_label = []
    test_mask = []

    for i in range(len(test_data)):
        test_img.append(test_data[i][0])
        test_label.append(test_data[i][1])
        test_mask.append(test_data[i][2])

    number_batch = len(test_data) // BATCH_SIZE

    smile_nb_true_pred = 0
    emotion_nb_true_pred = 0    
    gender_nb_true_pred = 0

    smile_nb_test = 0
    emotion_nb_test = 0
    gender_nb_test = 0

    for batch in range(number_batch):

        top = batch * BATCH_SIZE
        bot = min((batch + 1) * BATCH_SIZE, len(test_data))
        batch_img = np.asarray(test_img[top:bot])
        batch_label = np.asarray(test_label[top:bot])
        batch_mask = np.asarray(test_mask[top:bot])

        batch_img = input_data.random_crop(batch_img, (48, 48), 10)

        for i in range(BATCH_SIZE):
            if batch_mask[i] == 0.0:
                smile_nb_test += 1
            else:
                if batch_mask[i] == 1.0:
                    emotion_nb_test += 1
                else:
                    gender_nb_test += 1

        avg_ttl = []
        avg_rgl = []
        avg_smile_loss = []
        avg_emotion_loss = []
        avg_gender_loss = []

        ttl, sml, eml, gel, l2l = sess.run([loss, smile_loss, emotion_loss, gender_loss, l2_loss],
                                                  feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                             phase_test: True,
                                                             keep_prob: 1})

        smile_nb_true_pred += sess.run(smile_true_pred, feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                   phase_test: False,
                                                                   keep_prob: 1})

        emotion_nb_true_pred += sess.run(emotion_true_pred,
                                         feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                    phase_test: False,
                                                    keep_prob: 1})

        gender_nb_true_pred += sess.run(gender_true_pred,
                                        feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                   phase_test: False,
                                                   keep_prob: 1})

        avg_ttl.append(ttl)
        avg_smile_loss.append(sml)
        avg_emotion_loss.append(eml)
        avg_gender_loss.append(gel)
        avg_rgl.append(l2l)

    avg_smile_loss = np.average(avg_smile_loss)
    avg_emotion_loss = np.average(avg_emotion_loss)
    avg_gender_loss = np.average(avg_gender_loss)
    avg_rgl = np.average(avg_rgl)
    avg_ttl = np.average(avg_ttl)    

    smile_test_accuracy = smile_nb_true_pred * 1.0 / smile_nb_test
    emotion_test_accuracy = emotion_nb_true_pred * 1.0 / emotion_nb_test
    gender_test_accuracy = gender_nb_true_pred * 1.0 / gender_nb_test

    print('\n')

    print('Smile task test accuracy: ' + str(smile_test_accuracy * 100))
    print('Emotion task test accuracy: ' + str(emotion_test_accuracy * 100))
    print('Gender task test accuracy: ' + str(gender_test_accuracy * 100))

    print('Total loss: ' + str(avg_ttl) + '. L2-loss: ' + str(avg_rgl))
    print('Smile loss: ' + str(avg_smile_loss))
    print('Emotion loss: ' + str(avg_emotion_loss))
    print('Gender loss: ' + str(avg_gender_loss))