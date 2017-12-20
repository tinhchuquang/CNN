import numpy as np
import cv2
import input_data


DATA_FOLDER = '/home/tinh/CNN/m_CNN/multiltask/database/'

train = np.load(DATA_FOLDER + 'train_smile.npy')
test = np.load(DATA_FOLDER + 'test_gender.npy')

for i in range(15):
	# cv2.imshow('img',(train[i][0]))
	cv2.imshow('img1', train[i][0])
	cv2.imshow('img',(train[i][0] - 128) / 255.0)
	print(train[i][0])
	print(train[i][1])
	cv2.waitKey(0)


# train_data = np.load(DATA_FOLDER+'train_emotion.npy')
# for i in range(5):
#     cv2.imshow('img', (train_data[i][0])/255.0)
#     print(train_data[i][0])
#     print(train_data[i][1])
#     cv2.waitKey(0)



