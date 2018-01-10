import tensorflow as tf
import os
import numpy as np
from const import *
import DenseNet_load_data as load_data
import DenseNet

def log_loss_accuracy(summary_writer ,loss, accuracy, epoch, prefix):
    summary = tf.Summary(value=[
        tf.Summary.Value(
            tag='loss_%s' % prefix, simple_value= float(loss)),
        tf.Summary.Value(
            tag='accuracy_%s' % prefix, simple_value = float(accuracy)),
    ])
    summary_writer.add_summary(summary, epoch)



if __name__ == "__main__":
    sess = tf.InteractiveSession()
    global_step = tf.contrib.framework.get_or_create_global_step()

    x, y_, is_training = DenseNet.input()
    y_age_conv, accuracy = DenseNet.inference(x, y_, is_training)
    age_loss, l2_loss, total_loss = DenseNet.losses(y_age_conv, y_)
    train_step, learning_rate = DenseNet.train_step(total_loss, global_step)

    saver = tf.train.Saver(max_to_keep=max_to_keep)
    if not os.path.isfile(save_folder + "model.ckpt.index"):
        print("Create new model")
        sess.run(tf.global_variables_initializer())
        print("OK")
    else:
        print("Restoring existed model")
        saver.restore(sess, save_folder + "model.ckpt")
        print('OK')
        print(global_step.eval())


    writer = tf.summary.FileWriter
    summary_writer = writer(logs_folder)
    summary_writer.add_graph(sess.graph)

    number_train_image, number_test_image = load_data.global_image()

    num_batch = int(number_train_image) // batch_size
    current_epoch = int(global_step.eval() /num_batch)
    lnr = int_lnr

    for epoch in range(current_epoch+1, num_epoch+1):
        current_step = int(global_step.eval())
        print('Epoch:' + str(epoch))

        sum_loss = []
        sum_l2 = []
        sum_al = []
        sum_acc = []

        if lnr > 1e-4:
            if epoch < 5:
                lnr = 0.01
            elif epoch < 15:
                lnr = 0.001
            else:
                lnr = 0.0001
        print("learning rate:" + str(lnr))
        point = 0

        for batch in range(num_batch):

            data_train, point = load_data.get_train_image(point, batch_size)
            if len(data_train) < batch_size:
                print('Error data')
            np.random.shuffle(data_train)
            train_img = []
            train_label = []
            for i in range(len(data_train)):
                train_img.append(data_train[i][0])
                train_label.append(data_train[i][1])
            batch_img = np.asarray(train_img)
            batch_label = np.asarray(train_label)

            ttl, l2l, _, acc = sess.run([total_loss, l2_loss, train_step, accuracy],
                                        feed_dict={x:batch_img, y_:batch_label,
                                                   is_training:True, learning_rate:lnr})

            if epoch % 1000 == 0:
                print('Training on batch %s / %s. Loss: %s' % (str(batch +1), str(num_batch), ttl)+ '/r')
            sum_loss.append(ttl)
            sum_acc.append(acc)
            sum_l2.append(l2l)

        mean_loss = np.mean(sum_loss)
        mean_acc = np.mean(sum_acc)
        mean_l2 = np.mean(sum_l2)

        print('\nTraining loss: %f' %mean_loss)
        print('L2 loss: %f' % mean_l2)
        print('Accuracy train: %f' % mean_acc)

        saver.save(sess, save_folder+ 'model' + str(epoch) + '.ckpt')
        saver.save(sess, save_folder+ 'model.ckpt')
        log_loss_accuracy(summary_writer, loss=mean_loss, accuracy=mean_acc, epoch=epoch, prefix='train')

        point = 0
        num_batch_test = number_test_image // batch_size

        for batch in range(num_batch_test):
            data_valid, point = load_data.get_valid_image(point, batch_size)
            if len(data_valid) < batch_size:
                print('Error data')
            np.random.shuffle(data_valid)
            valid_img = []
            valid_label = []
            for i in range(len(data_valid)):
                valid_img.append(data_valid[i][0])
                valid_label.append(data_valid[i][1])
            batch_img = np.asarray(valid_img)
            batch_label = np.asarray(valid_label)

            ttl, l2l, _, acc = sess.run([total_loss, l2_loss, train_step, accuracy],
                                        feed_dict={x: batch_img, y_: batch_label,
                                                   is_training: True, learning_rate: lnr})

            sum_loss.append(ttl)
            sum_acc.append(acc)
            sum_l2.append(l2l)

        mean_loss = np.mean(sum_loss)
        mean_acc = np.mean(sum_acc)
        mean_l2 = np.mean(sum_l2)

        print('\nValid loss: %f' % mean_loss)
        print('L2 loss: %f' % mean_l2)
        print('Accuracy valid: %f' % mean_acc)





