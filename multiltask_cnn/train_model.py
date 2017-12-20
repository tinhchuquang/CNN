import layer
import input_data
import tensorflow as tf
import numpy as np
import os

NUMBER_SMILE_TEST = 1000
NUMBER_GENDER_TEST = 6455

USE_GPU = True
SAVE_FOLDER = '/home/tinh/CNN/m_CNN/multiltask/save_model'
NUM_EPOCHS = 2000
BATCH_SIZE = 64
NUM_DUPLICATE_SMILE = 9
EMOTION_IMAGE_FOR_TRAIN = NUM_DUPLICATE_SMILE * 3000
gender_IMAGE_FOR_TRAIN = NUM_DUPLICATE_SMILE * 3000
SMILE_SIZE = 96
EMOTION_SIZE = 48
GENDER_SIZE = 48
DROP_OUT_PROB = 0.5

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

''' PREPARE DATA '''
smile_train, smile_test = input_data.getSmileImage()
emotion_train, emotion_public_test, emotion_private_test = input_data.getEmotionImage()
gender_train, gender_test = input_data.getGenderImage()
'''--------------------------------------------------------------------------------------------'''


def one_hot(index, num_classes):
    assert index < num_classes and index >= 0
    tmp = np.zeros(num_classes)
    tmp[index] = 1
    return tmp

def eval_smile_public_test_emotion(nbof_crop):
    nbof_smile = len(smile_test)
    nbof_emotion = len(emotion_public_test)
    nbof_gender = len(gender_test)

    nbof_true_emotion = 0
    nbof_true_gender = 0
    nbof_true_smile = 0

    for i in range(nbof_emotion):
        emotion = np.zeros([1, 48, 48, 1])
        emotion[0] = emotion_public_test[i][0]
        emotion_label = np.argmax(emotion_public_test[i][1])

        smile = np.zeros([1, 96, 96, 1])
        smile[0] = smile_test[i % NUMBER_SMILE_TEST][0]
        smile_label = smile_test[i % NUMBER_SMILE_TEST][1]

        gender = np.zeros([1, 48, 48, 3])
        gender[0] = gender_test[i % NUMBER_GENDER_TEST][0]
        gender_label = (int)(gender_test[i % NUMBER_GENDER_TEST][1])

        y_emotion_pred = np.zeros([7])
        y_smile_pred = np.zeros([2])
        y_gender_pred = np.zeros([2])

        for _ in range(nbof_crop):
            x_emotion_ = input_data.augmentation_batch(emotion, (48, 48), 10)
            x_smile_ = input_data.augmentation_batch(smile, (96, 96), 10)
            x_gender_ = input_data.augmentation_batch(gender, (48, 48), 10)

            x_emotion_ = input_data.from_2d_to_3d(x_emotion_)
            x_smile_ = input_data.from_2d_to_3d(x_smile_)
            
            y1 = y_emotion_conv.eval(feed_dict={x_smile: x_smile_,
                                                x_emotion: x_emotion_,
                                                x_gender: x_gender_,
                                                keep_prob_smile_fc1: 1,
                                                keep_prob_smile_fc2: 1,
                                                keep_prob_emotion_fc1: 1,
                                                keep_prob_emotion_fc2: 1,
                                                keep_prob_gender_fc1: 1,
                                                keep_prob_gender_fc2: 1,
                                                is_training: False})
            y2 = y_smile_conv.eval(feed_dict={x_smile: x_smile_,
                                                x_emotion: x_emotion_,
                                                x_gender: x_gender_,
                                                keep_prob_smile_fc1: 1,
                                                keep_prob_smile_fc2: 1,
                                                keep_prob_emotion_fc1: 1,
                                                keep_prob_emotion_fc2: 1,
                                                keep_prob_gender_fc1: 1,
                                                keep_prob_gender_fc2: 1,
                                                is_training: False})

            y3 = y_gender_conv.eval(feed_dict={x_smile: x_smile_,
                                              x_emotion: x_emotion_,
                                              x_gender: x_gender_,
                                              keep_prob_smile_fc1: 1,
                                              keep_prob_smile_fc2: 1,
                                              keep_prob_emotion_fc1: 1,
                                              keep_prob_emotion_fc2: 1,
                                              keep_prob_gender_fc1: 1,
                                              keep_prob_gender_fc2: 1,
                                              is_training: False})
            y_emotion_pred += y1[0]
            y_smile_pred += y2[0]
            y_gender_pred += y3[0]

        predict_emotion = np.argmax(y_emotion_pred)
        predict_smile = np.argmax(y_smile_pred)
        predict_gender = np.argmax(y_gender_pred)

        if (predict_emotion == emotion_label):
            nbof_true_emotion += 1
        if (predict_smile == smile_label) & (i < NUMBER_SMILE_TEST):
            nbof_true_smile += 1
        if (predict_gender == gender_label) & (i < NUMBER_GENDER_TEST):
            nbof_true_gender += 1
    return nbof_true_smile * 100.0 / nbof_smile, nbof_true_emotion * 100.0 / nbof_emotion, nbof_true_gender * 100.0 / nbof_gender

def eval_smile_private_test_emotion(nbof_crop):
    nbof_smile = len(smile_test)
    nbof_emotion = len(emotion_private_test)
    nbof_gender = len(gender_test)

    nbof_true_emotion = 0
    nbof_true_smile = 0
    nbof_true_gender = 0

    for i in range(nbof_emotion):
        emotion = np.zeros([1, 48, 48, 1])
        emotion[0] = emotion_private_test[i][0]
        emotion_label = np.argmax(emotion_private_test[i][1])
        smile = np.zeros([1, 96, 96, 1])
        smile[0] = smile_test[i % NUMBER_SMILE_TEST][0]
        smile_label = smile_test[i % NUMBER_SMILE_TEST][1]

        gender = np.zeros([1, 48, 48, 3])
        gender[0] = gender_test[i % NUMBER_GENDER_TEST][0]
        gender_label = (int)(gender_test[i % NUMBER_GENDER_TEST][1])

        y_emotion_pred = np.zeros([7])
        y_smile_pred = np.zeros([2])
        y_gender_pred = np.zeros([2])

        for _ in range(nbof_crop):
            x_emotion_ = input_data.augmentation_batch(emotion, (48, 48), 10)
            x_smile_ = input_data.augmentation_batch(smile, (96, 96), 10)
            x_gender_ = input_data.augmentation_batch(gender, (48, 48), 10)

            x_emotion_ = input_data.from_2d_to_3d(x_emotion_)
            x_smile_ = input_data.from_2d_to_3d(x_smile_)

            y1 = y_emotion_conv.eval(feed_dict={x_smile: x_smile_,
                                                x_emotion: x_emotion_,
                                                x_gender: x_gender_,
                                                keep_prob_smile_fc1: 1,
                                                keep_prob_smile_fc2: 1,
                                                keep_prob_emotion_fc1: 1,
                                                keep_prob_emotion_fc2: 1,
                                                keep_prob_gender_fc1: 1,
                                                keep_prob_gender_fc2: 1,
                                                is_training: False})
            y2 = y_smile_conv.eval(feed_dict={x_smile: x_smile_,
                                              x_emotion: x_emotion_,
                                              x_gender: x_gender_,
                                              keep_prob_smile_fc1: 1,
                                              keep_prob_smile_fc2: 1,
                                              keep_prob_emotion_fc1: 1,
                                              keep_prob_emotion_fc2: 1,
                                              keep_prob_gender_fc1: 1,
                                              keep_prob_gender_fc2: 1,
                                              is_training: False})

            y3 = y_gender_conv.eval(feed_dict={x_smile: x_smile_,
                                               x_emotion: x_emotion_,
                                               x_gender: x_gender_,
                                               keep_prob_smile_fc1: 1,
                                               keep_prob_smile_fc2: 1,
                                               keep_prob_emotion_fc1: 1,
                                               keep_prob_emotion_fc2: 1,
                                               keep_prob_gender_fc1: 1,
                                               keep_prob_gender_fc2: 1,
                                               is_training: False})
            y_emotion_pred += y1[0]
            y_smile_pred += y2[0]
            y_gender_pred += y3[0]

        predict_emotion = np.argmax(y_emotion_pred)
        predict_smile = np.argmax(y_smile_pred)
        predict_gender = np.argmax(y_gender_pred)

        if (predict_emotion == emotion_label):
            nbof_true_emotion += 1
        if (predict_smile == smile_label) & (i < NUMBER_SMILE_TEST):
            nbof_true_smile += 1
        if (predict_gender == smile_label) & (i < NUMBER_GENDER_TEST):
            nbof_true_gender += 1
    return nbof_true_smile * 100.0 / nbof_smile, nbof_true_emotion * 100.0 / nbof_emotion, nbof_true_gender * 100.0 / nbof_gender

def evaluate(nbof_crop):
    print('Testing phase...............................')
    smile_acc, public_acc, gender_acc = eval_smile_public_test_emotion(nbof_crop)
    _, private_acc, _ = eval_smile_private_test_emotion(nbof_crop)
    print('Smile test accuracy: ', str(smile_acc))
    print('Gender test accuracy: ', str(gender_acc))
    print('Emotion public test accuracy: ', str(public_acc))
    print('Emotion private test accuracy: ', str(private_acc))


if __name__ == "__main__":
    if USE_GPU:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        sess = tf.InteractiveSession()
    global_step = tf.contrib.framework.get_or_create_global_step()

    if not os.path.isfile(SAVE_FOLDER + '/model.ckpt.index'):  # Create new model
        print('Create new model')
        x_smile, y_smile, x_emotion, y_emotion, x_gender, y_gender = layer.Input()
        y_smile_conv, y_emotion_conv, y_gender_conv = layer.inference(x_smile, x_emotion, x_gender)

        loss = layer.loss(y_smile_conv, y_smile, y_emotion_conv, y_emotion, y_gender_conv, y_gender)

        train_step = layer.train_op(loss, global_step)

        validation_acc = []
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
    else:
        print(SAVE_FOLDER + '/model.ckpt.index')
        print('Load exist model')
        saver = tf.train.import_meta_graph(SAVE_FOLDER + '/model.ckpt.meta')
        saver.restore(sess, SAVE_FOLDER + '/model.ckpt')
        print('Load OK')

    regul_loss = tf.get_collection('regul_loss')[0]
    total_loss = tf.get_collection('total_loss')[0]
    smile_loss = tf.get_collection('smile_loss')[0]
    emotion_loss = tf.get_collection('emotion_loss')[0]
    gender_loss = tf.get_collection('gender_loss')[0]

    learning_rate = tf.get_collection('learning_rate')[0]
    train_step = tf.get_collection('train_step')[0]

    x_smile = tf.get_collection('x_smile')[0]
    y_smile = tf.get_collection('y_smile')[0]
    x_emotion = tf.get_collection('x_emotion')[0]
    y_emotion = tf.get_collection('y_emotion')[0]
    x_gender = tf.get_collection('x_gender')[0]
    y_gender = tf.get_collection('y_gender')[0]

    keep_prob_smile_fc1 = tf.get_collection('keep_prob_smile_fc1')[0]
    keep_prob_emotion_fc1 = tf.get_collection('keep_prob_emotion_fc1')[0]
    keep_prob_smile_fc2 = tf.get_collection('keep_prob_smile_fc2')[0]
    keep_prob_emotion_fc2 = tf.get_collection('keep_prob_emotion_fc2')[0]
    keep_prob_gender_fc1 = tf.get_collection('keep_prob_gender_fc1')[0]
    keep_prob_gender_fc2 = tf.get_collection('keep_prob_gender_fc2')[0]

    y_smile_conv = tf.get_collection('y_smile_conv')[0]
    y_emotion_conv = tf.get_collection('y_emotion_conv')[0]
    y_gender_conv = tf.get_collection('y_gender_conv')[0]

    is_training = tf.get_collection('is_training')[0]
    loss_summary_placeholder = tf.get_collection('loss_summary_placeholder')[0]
    summary_op = tf.get_collection('summary_op')[0]
    train_writer = tf.summary.FileWriter('summary/train', sess.graph)

    smile_correct_prediction = tf.equal(tf.arg_max(y_smile_conv, 1), tf.arg_max(y_smile, 1))
    emotion_correct_prediction = tf.equal(tf.arg_max(y_emotion_conv, 1), tf.arg_max(y_emotion, 1))
    gender_correct_prediction = tf.equal(tf.arg_max(y_gender_conv, 1), tf.arg_max(y_gender, 1))

    smile_true_pred = tf.reduce_sum(tf.cast(smile_correct_prediction, dtype=tf.float32))
    emotion_true_pred = tf.reduce_sum(tf.cast(emotion_correct_prediction, dtype=tf.float32))
    gender_true_pred = tf.reduce_sum(tf.cast(gender_correct_prediction, dtype=tf.float32))

    for epoch in range(NUM_EPOCHS):

        smile_train_image = []
        smile_train_label = []

        emotion_train_image = []
        emotion_train_label = []

        gender_train_image = []
        gender_train_label = []

        # Shuffle data
        np.random.shuffle(smile_train)
        np.random.shuffle(emotion_train)
        np.random.shuffle(gender_train)

        for i in range(EMOTION_IMAGE_FOR_TRAIN):
            img = emotion_train[i % len(emotion_train)][0]
            label = emotion_train[i % len(emotion_train)][1]
            emotion_train_image.append(img)
            emotion_train_label.append(one_hot((int)(label), 7))

        for i in range(gender_IMAGE_FOR_TRAIN):
            img = gender_train[i % len(gender_train)][0]
            label = gender_train[i % len(gender_train)][1]
            gender_train_image.append(img)
            gender_train_label.append(one_hot((int)(label), 2))

        for _ in range(NUM_DUPLICATE_SMILE):
            np.random.shuffle(smile_train)
            for i in range(len(smile_train)):
                img = smile_train[i][0]
                label = smile_train[i][1]
                smile_train_image.append(img)
                smile_train_label.append(one_hot(label, 2))

        number_train = min(len(smile_train_image), len(emotion_train_image))

        avg_ttl = []
        avg_rgl = []
        avg_smile_loss = []
        avg_emotion_loss = []
        avg_gender_loss = []

        smile_nb_true_pred = 0
        emotion_nb_true_pred = 0
        gender_nb_true_pred = 0
        print("Epoch: %d" % epoch)
        print("Learning rate: %f" % learning_rate.eval())
        number_batch = number_train // BATCH_SIZE

        for i in range(number_batch):
            print('Training on batch '+ str(i + 1)+ '/'+ str(number_batch)+ '\r')
            top = i * BATCH_SIZE
            bot = min((i + 1) * BATCH_SIZE, number_train)
            x_smile_batch, y_smile_batch = smile_train_image[top:bot], smile_train_label[top:bot]
            x_emotion_batch, y_emotion_batch = emotion_train_image[top:bot], emotion_train_label[top:bot]
            x_gender_batch, y_gender_batch = gender_train_image[top:bot], gender_train_label[top:bot]

            x_smile_batch = input_data.augmentation_batch(x_smile_batch, SMILE_SIZE)
            x_emotion_batch = input_data.augmentation_batch(x_emotion_batch, EMOTION_SIZE)
            x_gender_batch = input_data.augmentation_batch(x_gender_batch, GENDER_SIZE)

            x_smile_batch = input_data.from_2d_to_3d(x_smile_batch)
            x_emotion_batch = input_data.from_2d_to_3d(x_emotion_batch)

            rgl, ttl, _, sm_loss, em_loss, ge_loss = sess.run(
                [regul_loss, total_loss, train_step, smile_loss, emotion_loss, gender_loss],
                feed_dict={x_smile: x_smile_batch, y_smile: y_smile_batch,
                           x_emotion: x_emotion_batch, y_emotion: y_emotion_batch,
                           x_gender: x_gender_batch, y_gender: y_gender_batch,
                           keep_prob_smile_fc1: 1 - DROP_OUT_PROB,
                           keep_prob_smile_fc2: 1 - DROP_OUT_PROB,
                           keep_prob_emotion_fc1: 1 - DROP_OUT_PROB,
                           keep_prob_emotion_fc2: 1 - DROP_OUT_PROB,
                           keep_prob_gender_fc1: 1 - DROP_OUT_PROB,
                           keep_prob_gender_fc2: 1 - DROP_OUT_PROB,
                           is_training: True})

            avg_rgl.append(rgl)
            avg_ttl.append(ttl)
            avg_smile_loss.append(sm_loss)
            avg_emotion_loss.append(em_loss)
            avg_gender_loss.append(ge_loss)

            smile_nb_true_pred += smile_true_pred.eval(feed_dict={x_smile: x_smile_batch, y_smile: y_smile_batch,
                                                                  x_emotion: x_emotion_batch,
                                                                  y_emotion: y_emotion_batch,
                                                                  x_gender: x_gender_batch, y_gender: y_gender_batch,
                                                                  keep_prob_smile_fc1: 1,
                                                                  keep_prob_smile_fc2: 1,
                                                                  keep_prob_emotion_fc1: 1,
                                                                  keep_prob_emotion_fc2: 1,
                                                                  keep_prob_gender_fc1: 1,
                                                                  keep_prob_gender_fc2: 1,
                                                                  is_training: False})

            emotion_nb_true_pred += emotion_true_pred.eval(feed_dict={x_smile: x_smile_batch, y_smile: y_smile_batch,
                                                                      x_emotion: x_emotion_batch,
                                                                      y_emotion: y_emotion_batch,
                                                                      x_gender: x_gender_batch,
                                                                      y_gender: y_gender_batch,
                                                                      keep_prob_smile_fc1: 1,
                                                                      keep_prob_smile_fc2: 1,
                                                                      keep_prob_emotion_fc1: 1,
                                                                      keep_prob_emotion_fc2: 1,
                                                                      keep_prob_gender_fc1: 1,
                                                                      keep_prob_gender_fc2: 1,
                                                                      is_training: False})

            gender_nb_true_pred += gender_true_pred.eval(feed_dict={x_smile: x_smile_batch, y_smile: y_smile_batch,
                                                                    x_emotion: x_emotion_batch,
                                                                    y_emotion: y_emotion_batch,
                                                                    x_gender: x_gender_batch,
                                                                    y_gender: y_gender_batch,
                                                                    keep_prob_smile_fc1: 1,
                                                                    keep_prob_smile_fc2: 1,
                                                                    keep_prob_emotion_fc1: 1,
                                                                    keep_prob_emotion_fc2: 1,
                                                                    keep_prob_gender_fc1: 1,
                                                                    keep_prob_gender_fc2: 1,
                                                                    is_training: False})

        sum_rgl = np.average(avg_rgl)
        sum_ttl = np.average(avg_ttl)

        sum_sm_loss = np.average(avg_smile_loss)
        sum_em_loss = np.average(avg_emotion_loss)
        sum_ge_loss = np.average(avg_gender_loss)

        smile_train_accuracy = smile_nb_true_pred * 1.0 / number_train
        emotion_train_accuracy = emotion_nb_true_pred * 1.0 / number_train
        gender_train_accuracy = gender_nb_true_pred * 1.0 / number_train
        print('Total loss: ' + str(sum_ttl) + '. L2-loss: ' + str(sum_rgl))
        print('Smile loss: ' + str(sum_sm_loss))
        print('Emotion loss: ' + str(sum_em_loss))
        print('Gender loss: ' + str(sum_ge_loss))
        print('Smile task train accuracy: ' + str(smile_train_accuracy * 100))
        print('Emotion task train accuracy: ' + str(emotion_train_accuracy * 100))
        print('Gender task train accuracy: ' + str(gender_train_accuracy * 100))

        #summary = sess.run(summary_op, feed_dict={loss_summary_placeholder: sum_ttl})
        #train_writer.add_summary(summary, global_step=epoch)

        if (epoch % 10 == 0) & (epoch != 0):
            print('Save model............................')
            saver.save(sess, SAVE_FOLDER + "/model.ckpt")
            print('Done!')
            evaluate(nbof_crop = 1)