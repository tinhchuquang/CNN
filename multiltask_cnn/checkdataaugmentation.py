import input_data
import os
import tensorflow as tf
import numpy as np
import NetModel
from const import *
import cv2


# smile_train, smile_test = input_data.getSmileImage()
emotion_train, emotion_public_test, emotion_private_test = input_data.getEmotionImage()
# gender_train, gender_test = input_data.getGenderImage()
label_emotion = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
label_smile = ['nosmile', 'smile']
label_gender = ['female','male']


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

    test_data=[]

    for i in range(len(emotion_private_test)):
        test_data.append((emotion_public_test[i][0], emotion_public_test[i][1], 1.0))
    np.random.shuffle(test_data)


    test_img = []
    test_label = []
    test_mask = []

    for i in range(len(test_data)):
        test_img.append(test_data[i][0])
        test_label.append(test_data[i][1])
        test_mask.append(test_data[i][2])



    print('Restore model')
    saver = tf.train.Saver()
    saver.restore(sess, SAVE_FOLDER2 + 'model.ckpt')

    print('OK')

    batch_img = np.asarray(test_img[0:BATCH_SIZE])
    batch_label = np.asarray(test_label[0:BATCH_SIZE])
    batch_mask = np.asarray(test_mask[0:BATCH_SIZE])

    print(batch_img.shape)
    

    emotion, smile, gender = sess.run([y_emotion_conv,y_smile_conv, y_gender_conv], feed_dict={x:batch_img, y_: batch_label, mask: batch_mask,
                                                                   phase_test: False,
                                                                   keep_prob: 1})

    for i in range(BATCH_SIZE):
        cv2.imshow("result", batch_img[i])
        print(label_emotion[np.argmax(emotion[i])])
        print(label_smile[np.argmax(smile[i])])
        print(label_gender[np.argmax(gender[i])])
        cv2.waitKey(0)

