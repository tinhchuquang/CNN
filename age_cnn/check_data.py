import cv2
import numpy as np 
import input_data


AGE_FOLDER = '/database/'
train_age, test_age = input_data.getImdbAgeImage()


def check_data_age():
	for i in range(5):
		cv2.imshow('img0', (train_age[i][0]) )
		print(train_age[i][0].shape)
		cv2.imshow('img', (train_age[i][0]-127) /255.0 )
		cv2.waitKey(0)


check_data_age()