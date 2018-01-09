import input_model as input_data
import os 
import tensorflow as tf 
import numpy as np 
import gender_model as netmodel 
from const import *


def one_hot(index, num_classes):
    assert index < num_classes and index > 0
    tmp = np.zeros(num_classes, dtype=np.float32)
    tmp[index] = 1.0 
    return tmp

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    global_step = tf.contrib.framework.get_or_create_global_step()

    x, y_ = netmodel.Input()
    y_gender_conv, phase_train, keep_prob = netmodel.NetModelLayer(x)
    gender_loss, l2_loss, total_loss = netmodel._cross_entropy_loss(y_gender_conv, y_)

    train_step , learning_rate = netmodel.train_op(total_loss, global_step)

    y_gender_conv = tf.nn.softmax(y_gender_conv)
    gender_correct_prediction = tf.equal(tf.argmax(y_gender_conv, 1), tf.argmax(y_, 1))
    gender_true_pred = tf.reduce_sum(tf.cast(gender_correct_prediction, dtype=tf.float32))

    saver = tf.train.Saver()

    if not os.path.isfile(SAVE_FOLDER + 'model.ckpt.index'):
        print('Create new model')
        sess.run(tf.global_variables_initializer())
        print('OK')
    else:
        print('Restoring exitsted model')
        saver.restore(sess, SAVE_FOLDER + 'model.ckpt')
        print('OK')

    loss_summary_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('loss', loss_summary_placeholder)
    accuracy_train = tf.placeholder(tf.float32)
    tf.summary.scalar('accuracy_train', accuracy_train)

    merge_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./summary/')
    writer.add_graph(sess.graph)

    number_batch = NUMBER_TRAIN_DATA  // BATCH_SIZE 
    current_epoch = (int)(global_step.eval() / number_batch)

    lr = 0.0001
    input_data.global_image()
    for epoch in range(current_epoch, NUM_EPOCHS):
        print('Epoch:', str(epoch))

        sum_loss = []
        sum_l2 = []
        sum_gl = []
        sum_acc = []

        print("learning rate:" + str(lr))
        point = 0
        for batch in range(number_batch):
            if batch%1000 == 0:
                print('Trainning on batch '+str(batch+1)+' / '+str(number_batch)+ '\r')
            # top = batch * BATCH_SIZE
            # bot = min((batch+1)*BATCH_SIZE, NUMBER_TRAIN_DATA)
            data_train, point = input_data.get_train_image(point, BATCH_SIZE)
            np.random.shuffle(data_train)
            train_img = []
            train_label = []
            for i in range(len(data_train)):
                train_img.append(data_train[i][0])
                train_label.append(data_train[i][1])
            batch_img = np.asarray(train_img)
            batch_label = np.asarray(train_label)

            # batch_img =  input_data.augmentation(batch_img, (112, 90))

            ttl, l2l, gl, _, acc = sess.run([total_loss, l2_loss, gender_loss, train_step, gender_true_pred],feed_dict={x:batch_img, y_: batch_label, phase_train: True,learning_rate: lr, keep_prob: 0.5})
            sum_loss.append(ttl)
            sum_acc.append(acc)
            sum_gl.append(gl)
            sum_l2.append(l2l)

        sum_loss = np.average(sum_loss)
        sum_gl = np.average(sum_gl)
        sum_l2 = np.average(sum_l2)
        sum_acc = np.average(sum_acc)

        print('Summary train')
        print('Total loss: '+ str(sum_loss)+ ', Gender loss: '+ str(sum_gl)+ ', L2 loss: '+ str(sum_l2))
        print('Gender train accuracy: '+ str(sum_acc))

        saver.save(sess, SAVE_FOLDER + 'model.ckpt')
        print("Save model sucess")

        if epoch % 1 == 0:
            point = 0
            number_batch_test = NUMBER_TEST_DATA // BATCH_SIZE
            sum_loss = []
            sum_l2 = []
            sum_gl = []
            sum_acc = []

            for batch in range(number_batch_test):
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



