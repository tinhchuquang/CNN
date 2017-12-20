import input_data 
import os 
import tensorflow as tf 
import numpy as np 
import netmodel as NetModel 

from const import *

''' PREPARE DATA '''
smile_train, smile_test = input_data.getSmileImage()
emotion_train, emotion_public_test, emotion_private_test = input_data.getEmotionImage()
gender_train, gender_test = input_data.getGenderImage()
age_train, age_test = input_data.getAgeImage()

'''---------------------------------------------------------------------------------------'''

def one_hot(index, num_classes):
	assert index < num_classes and index >= 0
	tmp = np.zeros(num_classes, dtype=np.float32)
	tmp[index] = 1.0
	return tmp

if __name__ == "__main__":
	sess = tf.InteractiveSession()
	global_step = tf.contrib.framework.get_or_create_global_step()
	x, y_, mask = NetModel.Input()
	y_smile_conv, y_emotion_conv, y_gender_conv, y_age_conv, phase_train, keep_prob = NetModel.NetmodelLayer(x)
	smile_loss, emotion_loss, gender_loss, age_loss, l2_loss, loss = NetModel._cross_entropy_loss(y_smile_conv, y_emotion_conv
		, y_gender_conv, y_age_conv, y_, mask)
	train_step = NetModel.train_op(loss, global_step)

	smile_mask = tf.get_collection('smile_mask')[0]
	emotion_mask = tf.get_collection('emotion_mask')[0]
	gender_mask = tf.get_collection('gender_mask')[0]
	age_mask = tf.get_collection('age_mask')[0]

	y_smile = tf.get_collection('y_smile')[0]
	y_emotion = tf.get_collection('y_emotion')[0]
	y_gender = tf.get_collection('y_gender')[0]
	y_age = tf.get_collection('y_age')[0]

	smile_correct_prediction = tf.equal(tf.argmax(y_smile_conv, 1), tf.argmax(y_smile, 1))
	emotion_correct_prediction = tf.equal(tf.argmax(y_emotion_conv, 1), tf.argmax(y_emotion, 1))
	gender_correct_prediction = tf.equal(tf.argmax(y_gender_conv, 1), tf.argmax(y_gender, 1))
	age_correct_prediction = tf.equal(tf.argmax(y_age_conv, 1), tf.argmax(y_age, 1))

	smile_true_pred = tf.reduce_sum(tf.cast(smile_correct_prediction, dtype=tf.float32) * smile_mask)
    emotion_true_pred = tf.reduce_sum(tf.cast(emotion_correct_prediction, dtype=tf.float32) * emotion_mask)
    gender_true_pred = tf.reduce_sum(tf.cast(gender_correct_prediction, dtype=tf.float32) * gender_mask)
    age_true_pred = tf.reduce_sum(tf.cast(age_correct_prediction, dtype=tf.float32)*age_mask)

    train_data = []
    #Mask: Smile -> 0, Emotion -> 1, Gender -> 2, Age -> 3
	for i in range(len(smile_train) * NUMBER_DUPLICATE_SMILE):
		img = (smile_train[i%3000][0] -128) / 255.0
		label = smile_train[i%3000][1]
		train_data.append((img, one_hot(label, 7), 0.0))
	for i in range(len(emotion_train)):
		train_data.append(emotion_train[i][0], emotion_train[i][1]. 1.0)
	for i in range(len(gender_train)):
		img = (gender_train[i][0] - 128) / 255.0
		label = gender_train[i][1]
		train_data.append((img, one_hot(label, 7), 2.0))
	for i in range(len(age_train)):
		img = age_train[i][0]-128 / 255.0
		label = gender_train[i][1]
		train_data.append((img, one_hot(label, 7), 3.0))

	np.random.shuffle(train_data)

	test_data = []
	#Mask Smile -> 0, Emotion-> 1, Gender -> 2, Age -> 3
	for i in range(len(smile_test) * NUMBER_DUPLICATE_SMILE):
		img = (smile_test[i%3000][0] -128) / 255.0
		label = smile_test[i%3000][1]
		test_data.append((img, one_hot(label, 7), 0.0))
	for i in range(len(emotion_test)):
		test_data.append(emotion_public_test[i][0], emotion_public_test[i][1]. 1.0)
	for i in range(len(gender_test)):
		img = (gender_test[i][0] - 128) / 255.0
		label = gender_test[i][1]
		test_data.append((img, one_hot(label, 7), 2.0))
	for i in range(len(age_test)):
		img = age_test[i][0]-128 / 255.0
		label = gender_test[i][1]
		test_data.append((img, one_hot(label, 7), 3.0))

	np.random.shuffle(test_data)

	saver = tf.train.Saver()

	if not os.path.isfile(SAVE_FORDER+ 'model.cpkt.index'):
		print('Create new model')
		sess.run(tf.global_variables_initializer())
		print('OK')
	else :
		print('Restoring existed model')
		saver.restore(sess, SAVE_FORDER+ 'model.cpkt')
		print('OK')

	loss_summary_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('loss', loss_summary_placeholder)
    accuracy_train = tf.placeholder(tf.float32)
    tf.summary.scalar('accuracy train', accuracy_train)

    # loss_summary_test_placeholder = tf.placeholder(tf.float32)
    # tf.summary.scalar('loss_test', loss_summary_test_placeholder)
    # accuracy_test = tf.placeholder(tf.float32)
    # tf.summary.scalar('accuracy test', accuracy_test)


    merge_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./summary/")
    # test_writer = tf.summary.FileWriter("./summary_test/")
    writer.add_graph(sess.graph)

    learning_rate = tf.get_collection('learning_rate')[0]

    current_epoch = (int)(global_step.eval() / (len(train_data) // BATCH_SIZE))
    for epoch in range(current_epoch + 1, NUM_EPOCHS):
        print('Epoch:', str(epoch))
        np.random.shuffle(train_data)
        train_img = []
        train_label = []
        train_mask = []

        for i in range(len(train_data)):
            train_img.append(train_data[i][0])
            train_label.append(train_data[i][1])
            train_mask.append(train_data[i][2])

        number_batch = len(train_data) // BATCH_SIZE

        avg_ttl = []
        avg_rgl = []
        avg_smile_loss = []
        avg_emotion_loss = []
        avg_gender_loss = []
        avg_age_loss = []

        smile_nb_true_pred = 0
        emotion_nb_true_pred = 0
        gender_nb_true_pred = 0
        age_nb_true_pred = 0

        smile_nb_train = 0
        emotion_nb_train = 0
        gender_nb_train = 0
        age_nb_train = 0
        print("Learning rate: %f" % learning_rate.eval())
        for batch in range(number_batch):
            print('Training on batch '+ str(batch + 1)+ '/'+ str(number_batch)+'\r')
            top = batch * BATCH_SIZE
            bot = min((batch + 1) * BATCH_SIZE, len(train_data))
            batch_img = np.asarray(train_img[top:bot])
            batch_label = np.asarray(train_label[top:bot])
            batch_mask = np.asarray(train_mask[top:bot])

            for i in range(BATCH_SIZE):
                if batch_mask[i] == 0.0:
                    smile_nb_train += 1
                else:
                    if batch_mask[i] == 1.0:
                        emotion_nb_train += 1
                    	if batch_mask[i] == 2.0
                        	gender_nb_train += 1
                    	else:
                    		age_nb_train += 1

            batch_img = input_data.augmentation(batch_img, 48)
            

            ttl, sml, eml, gel, l2l, al, _ = sess.run([loss, smile_loss, emotion_loss, gender_loss, age_loss, l2_loss, train_step],
                                                  feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                             phase_train: True,
                                                             keep_prob: 0.5})

            smile_nb_true_pred += sess.run(smile_true_pred, feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                       phase_train: True,
                                                                       keep_prob: 0.5})

            emotion_nb_true_pred += sess.run(emotion_true_pred,
                                             feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                        phase_train: True,
                                                        keep_prob: 0.5})

            gender_nb_true_pred += sess.run(gender_true_pred,
                                            feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                       phase_train: True,
                                                       keep_prob: 0.5})

            age_nb_true_pred += sess.run(age_true_pred, 
            							 feed_dict={x: batch_img, y_: batch_label, mask: batch_mask, 
            							 			phase_train:True,
            							 			keep_prob: 0.5})


            avg_ttl.append(ttl)
            avg_smile_loss.append(sml)
            avg_emotion_loss.append(eml)
            avg_gender_loss.append(gel)
            avg_age_loss.append(al)
            avg_rgl.append(l2l)

        smile_train_accuracy = smile_nb_true_pred * 1.0 / smile_nb_train
        emotion_train_accuracy = emotion_nb_true_pred * 1.0 / emotion_nb_train
        gender_train_accuracy = gender_nb_true_pred * 1.0 / gender_nb_train
        age_train_accuracy = age_nb_true_pred * 1.0 / age_nb_train

        avg_smile_loss = np.average(avg_smile_loss)
        avg_emotion_loss = np.average(avg_emotion_loss)
        avg_gender_loss = np.average(avg_gender_loss)
        avg_age_loss = np.average(avg_age_loss)
        avg_rgl = np.average(avg_rgl)
        avg_ttl = np.average(avg_ttl)

        accuracy_train_result=(smile_train_accuracy+emotion_train_accuracy+gender_train_accuracy + avg_age_loss)/4
        
        summary = sess.run(merge_summary, feed_dict={loss_summary_placeholder: avg_ttl, accuracy_train:accuracy_train_result})
        writer.add_summary(summary, global_step=epoch)

        print('\nTrain data')

        print('Smile task train accuracy: ' + str(smile_train_accuracy * 100))
        print('Emotion task train accuracy: ' + str(emotion_train_accuracy * 100))
        print('Gender task train accuracy: ' + str(gender_train_accuracy * 100))
        print('Age task train accuracy: ' + str(age_train_accuracy * 100))
        print('Total loss: ' + str(avg_ttl) + '. L2-loss: ' + str(avg_rgl))
        print('Smile loss: ' + str(avg_smile_loss))
        print('Emotion loss: ' + str(avg_emotion_loss))
        print('Gender loss: ' + str(avg_gender_loss))
        print('Age loss: ' + str(avg_age_loss))

        saver.save(sess, SAVE_FOLDER2 + 'model.ckpt')
        print("Save model sucess")



        if epoch%10 == 0:
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
            age_nb_true_pred = 0

            smile_nb_test = 0
            emotion_nb_test = 0
            gender_nb_test = 0
            age_nb_test = 0

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
                avg_age_loss = []


                ttl, sml, eml, gel, al,  l2l = sess.run([loss, smile_loss, emotion_loss, gender_loss, age_loss, l2_loss],
                                                          feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                     phase_train: False,
                                                                     keep_prob: 1})

                smile_nb_true_pred += sess.run(smile_true_pred, feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                           phase_train: False,
                                                                           keep_prob: 1})

                emotion_nb_true_pred += sess.run(emotion_true_pred,
                                                 feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                            phase_train: False,
                                                            keep_prob: 1})

                gender_nb_true_pred += sess.run(gender_true_pred,
                                                feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                           phase_train: False,
                                                           keep_prob: 1})

                age_nb_true_pred += sess.run(age_true_pred,
                                                feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                           phase_train: False,
                                                           keep_prob: 1})


                avg_ttl.append(ttl)
                avg_smile_loss.append(sml)
                avg_emotion_loss.append(eml)
                avg_gender_loss.append(gel)
                avg_age_loss.append(al)
                avg_rgl.append(l2l)

            avg_smile_loss = np.average(avg_smile_loss)
            avg_emotion_loss = np.average(avg_emotion_loss)
            avg_gender_loss = np.average(avg_gender_loss)
            avg_rgl = np.average(avg_rgl)
            avg_ttl = np.average(avg_ttl)    

            smile_test_accuracy = smile_nb_true_pred * 1.0 / smile_nb_test
            emotion_test_accuracy = emotion_nb_true_pred * 1.0 / emotion_nb_test
            gender_test_accuracy = gender_nb_true_pred * 1.0 / gender_nb_test
            age_test_accuracy = age_nb_true_pred *1.0 / age_nb_test

            # accuracy_test_result = (smile_test_accuracy + emotion_test_accuracy+gender_test_accuracy)/3

            # summary = sess.run(merge_summary, feed_dict={loss_summary_placeholder: avg_ttl, accuracy_test:accuracy_test_result})
            # writer.add_summary(summary, global_step=epoch)

            print('\nTest')

            print('Smile task test accuracy: ' + str(smile_test_accuracy * 100))
            print('Emotion task test accuracy: ' + str(emotion_test_accuracy * 100))
            print('Gender task test accuracy: ' + str(gender_test_accuracy * 100))
            print('Age task test accuracy: ' + str(age_test_accuracy * 100))

            print('Total loss: ' + str(avg_ttl) + '. L2-loss: ' + str(avg_rgl))
            print('Smile loss: ' + str(avg_smile_loss))
            print('Emotion loss: ' + str(avg_emotion_loss))
            print('Gender loss: ' + str(avg_gender_loss))
            print('Age loss: ' + str(avg_age_loss))





        
