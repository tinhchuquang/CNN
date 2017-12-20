import input_data
import os
import tensorflow as tf
import numpy as np
import layer as NetModel
from const import *

''' PREPARE DATA '''
train_data, test_data = input_data.getImdbImage()
'''--------------------------------------------------------------------------------------------'''


def one_hot(index, num_classes):
    assert index < num_classes and index >= 0
    tmp = np.zeros(num_classes, dtype=np.float32)
    tmp[index] = 1.0
    return tmp


if __name__ == "__main__":
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8

    sess = tf.InteractiveSession()
    global_step = tf.contrib.framework.get_or_create_global_step()

    x, y_= NetModel.Input()

    y_age_conv, phase_train, keep_prob = NetModel.NetModelLayer(x)

    age_loss, l2_loss, loss = NetModel._cross_entropy_loss(y_age_conv, y_)

    train_step = NetModel.train_op(loss, global_step)

    age_correct_prediction = tf.equal(tf.argmax(y_age_conv, 1), tf.argmax(y_, 1))
    age_true_pred = tf.reduce_sum(tf.cast(age_correct_prediction, dtype=tf.float32))


    

    # train_data = []
    
    # for i in range(len(train_data_age)):
    #     img = (train_data_age[i][0] - 128) / 255.0
    #     #img = (gender_train[i][0]) 
    #     label = (int)(train_data_age[i][1])
    #     train_data.append((img, one_hot(label, 13)))

    saver = tf.train.Saver()

    if not os.path.isfile(SAVE_FOLDER + 'model.ckpt.index'):
        print('Create new model')
        sess.run(tf.global_variables_initializer())
        print('OK')
    else:
        print('Restoring existed model')
        saver.restore(sess, SAVE_FOLDER + 'model.ckpt')
        print('OK')

    loss_summary_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('loss', loss_summary_placeholder)
    accuracy_train = tf.placeholder(tf.float32)
    tf.summary.scalar('accuracy_train', accuracy_train)

    merge_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./summary/")
    writer.add_graph(sess.graph)

    learning_rate = tf.get_collection('learning_rate')[0]

    current_epoch = (int)(global_step.eval() / (len(train_data) // BATCH_SIZE))
    for epoch in range(current_epoch + 1, NUM_EPOCHS):
        print('Epoch:', str(epoch))
        np.random.shuffle(train_data)
        train_img = []
        train_label = []

        for i in range(len(train_data)):
            train_img.append((train_data[i][0]-128) / 255.0) 
            train_label.append(train_data[i][1])

        number_batch = len(train_data) // BATCH_SIZE

        age_nb_true_pred = 0
        avg_ttl = []
        avg_rgl = []
        avg_age_loss = []

        print("Learning rate: %f" % learning_rate.eval())
        for batch in range(number_batch):
            print('Training on batch '+ str(batch + 1)+ '/'+ str(number_batch)+'\r')
            top = batch * BATCH_SIZE
            bot = min((batch + 1) * BATCH_SIZE, len(train_data))
            batch_img = np.asarray(train_img[top:bot])
            batch_label = np.asarray(train_label[top:bot])

            batch_img = input_data.augmentation(batch_img, 48)
            ttl, ageloss, l2l, _ = sess.run([loss, age_loss, l2_loss, train_step],
                                                  feed_dict={x: batch_img, y_: batch_label,
                                                             phase_train: True,
                                                             keep_prob: 0.5})

            age_nb_true_pred += sess.run(age_true_pred, feed_dict={x: batch_img, y_: batch_label,
                                                                       phase_train: True,
                                                                       keep_prob: 0.5})


            avg_ttl.append(ttl)
            avg_age_loss.append(ageloss)
            avg_rgl.append(l2l)

        age_train_accuracy = age_nb_true_pred * 1.0 / 150000
        avg_age_loss = np.average(avg_age_loss)
        avg_rgl = np.average(avg_rgl)
        avg_ttl = np.average(avg_ttl)


        summary = sess.run(merge_summary, feed_dict={loss_summary_placeholder: avg_ttl, accuracy_train: age_train_accuracy})
        writer.add_summary(summary, global_step=epoch)

        print('\n')

        print('Age train accuracy: ' + str(age_train_accuracy * 100))
        print('Total loss: ' + str(avg_ttl) + '. L2-loss: ' + str(avg_rgl))
        print('Age loss: ' + str(avg_age_loss))

        saver.save(sess, SAVE_FOLDER + 'model.ckpt')
        print('SAVE DONE!')

        test_img = []
        test_label = []


        for i in range(len(test_data)):
            test_img.append((test_data[i][0]-128) / 255.0)
            test_label.append(test_data[i][1])

        if epoch%1 == 0:

            number_batch = len(test_data) // BATCH_SIZE
            age_nb_true_pred = 0

            for batch in range(number_batch):
                top = batch * BATCH_SIZE
                bot = min((batch +1)*BATCH_SIZE, len(test_data))
                batch_img = np.asarray(test_img[top:bot])
                batch_label = np.asarray(test_label[top:bot])

                batch_img = input_data.random_crop(batch_img, (48, 48), 10)

                avg_ttl = []
                avg_rgl = []
                avg_age_loss = []

                ttl, age_loss, l2l, _ = sess.run([loss, age_loss, l2_loss, train_step], 
                                                feed_dict={x:batch_img, y_:batch_label,
                                                phase_train:False,
                                                keep_prob:1.0})
                age_nb_true_pred += sess.run(age_true_pred, feed_dict={x:batch_img, y_:batch_label, 
                                            phase_train:False, 
                                            keep_prob:1.0})


                avg_ttl.append(ttl)
                avg_age_loss.append(ageloss)
                avg_rgl.append(l2l)

            age_train_accuracy = age_nb_true_pred * 1.0 / 150000
            avg_age_loss = np.average(avg_age_loss)
            avg_rgl = np.average(avg_rgl)
            avg_ttl = np.average(avg_ttl)

            print('Age test accuracy: ' + str(age_train_accuracy * 100))
            print('Total loss: ' + str(avg_ttl) + '. L2-loss: ' + str(avg_rgl))
            print('Age loss: ' + str(avg_age_loss))