import layer
import input_data
import tensorflow as tf 
import numpy as np
import os 
import cv2

NUMBER_SMILE_TEST = 1000
NUMBER_GENDER_TEST = 6455

USE_GPU = True
SAVE_FOLDER = '/home/tinh/CNN/m_CNN/multiltask/save_model'
NUM_EPOCHS = 2000
BATCH_SIZE = 128
NUM_DUPLICATE_SMILE = 9
EMOTION_IMAGE_FOR_TRAIN = NUM_DUPLICATE_SMILE* 3000
gender_IMAGE_FOR_TRAIN = NUM_DUPLICATE_SMILE* 3000
SMILE_SIZE = 48
EMOTION_SIZE = 48
GENDER_SIZE = 48
DROP_OUT_PROB = 0.5

#import data
smile_train, smile_test = input_data.getSmileImage()
emotion_train, emotion_public_test, emotion_private_test = input_data.getEmotionImage()
gender_train, gender_test = input_data.getGenderImage()


def one_hot(index, num_classes):
	assert index < num_classes and index >= 0
	tmp = np.zeros(num_classes)
	tmp[index] = 1
	return tmp


def run():
	train_data = []
    # Mask: Smile -> 0, Emotion -> 1, Gender -> 2
	for i in range(len(smile_train) * 10):
		img = (smile_train[i % 3000][0] - 128) / 255.0
		label = smile_train[i % 3000][1]
		train_data.append((img, one_hot(label, 7), 0.0))
	for i in range(len(emotion_train)):
		train_data.append((emotion_train[i][0], one_hot(emotion_train[i][1], 7), 1.0))
	for i in range(len(gender_train)):
		img = (gender_train[i][0] - 128) / 255.0
		label = (int)(gender_train[i][1])
		train_data.append((img, one_hot(label, 7), 2.0))

	print(len(train_data))	

	# current_epoch = (len(train_data) // BATCH_SIZE)
	# for epoch in range(current_epoch + 1, NUM_EPOCHS):
	# 	print('Epoch:', str(epoch))
	# 	np.random.shuffle(train_data)
	# 	train_img = []
	# 	train_label = []
	# 	train_mask = []

	# 	for i in range(len(train_data)):
	# 		train_img.append(train_data[i][0])
	# 		train_label.append(train_data[i][1])
	# 		train_mask.append(train_data[i][2])

	# 	number_batch = len(train_data) // BATCH_SIZE

	# 	avg_ttl = []
	# 	avg_rgl = []
	# 	avg_smile_loss = []
	# 	avg_emotion_loss = []
	# 	avg_gender_loss = []

	# 	smile_nb_true_pred = 0
	# 	emotion_nb_true_pred = 0
	# 	gender_nb_true_pred = 0

	# 	smile_nb_train = 0
	# 	emotion_nb_train = 0
	# 	gender_nb_train = 0

	# 	for batch in range(number_batch):
	# 		print('Training on batch '+ str(batch + 1)+ '/'+ str(number_batch)+'\r')
	# 		top = batch * BATCH_SIZE
	# 		bot = min((batch + 1) * BATCH_SIZE, len(train_data))
	# 		batch_img = np.asarray(train_img[top:bot])
	# 		batch_label = np.asarray(train_label[top:bot])
	# 		batch_mask = np.asarray(train_mask[top:bot])

	# 		for i in range(BATCH_SIZE):
	# 			if batch_mask[i] == 0.0:
	# 				smile_nb_train += 1
	# 			else:
	# 				if batch_mask[i] == 1.0:
	# 					emotion_nb_train += 1
	# 				else:
	# 					gender_nb_train += 1

	# 		batch_img = input_data.augmentation_batch(batch_img, 48)
	# 		batch_img = input_data.from_2d_to_3d(batch_img)

run()