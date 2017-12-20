import numpy as np 
import cv2

SMILE_DATA =  '/home/tinh/CNN/m_CNN/SMILE_DATA/'
EMOTION_LABEL = '/home/tinh/CNN/fer2013/fer2013.csv'
EMOTION_IMAGE_TEST  = '/home/tinh/CNN/fer2013/test/'
EMOTION_IMAGE_TRAIN = '/home/tinh/CNN/fer2013/train/'


def load_data(SAVE_FOLDER):
	all_image = []
	all_label = np.zeros((4000, 1))
	
	for i in range(4000):
		name_image = SAVE_FOLDER
		if i < 9:
			name_image = name_image + str(i+1) +'.jpg'
			all_image.append(cv2.imread(name_image, 0))
		elif i < 99:
			name_image = name_image +str(i+1) +'.jpg'
			all_image.append(cv2.imread(name_image, 0))
		elif i < 999:
			name_image = name_image + str(i+1) + '.jpg'
			all_image.append(cv2.imread(name_image, 0))
		else:
			name_image = name_image +str(i+1) +'.jpg'
			all_image.append(cv2.imread(name_image, 0))
	labels = open('/home/tinh/CNN/m_CNN/GENKI4K/labels.txt', 'r')
	index = 0
	for line in labels:
		all_label[index] = int(line[0])
		index+= 1
	all_label = np.asarray(all_label, dtype='int32')
	#shuffle data
	index = np.arange(4000)
	np.random.shuffle(index)
	train = index[0:3000]
	test = index[3000:]
	y_train = []
	y_test = []
	X_train = []
	X_test = []
	for i in range(4000):
		if i < 3000:
			X_train.append(all_image[index[i]])
			y_train.append(all_label[index[i]])
		else :
			X_test.append(all_image[index[i]])
			y_test.append(all_label[index[i]])

	return X_train, y_train, X_test, y_test

def load_csv(path):
	data = pd.read_csv(path)
	label_train = []
	pixel_train = []
	label_test = []
	pixel_test = []
	label_val= []
	pixel_val = []
	for i in range(len(data)):
		if 'Training' in data['Usage'][i]:
			label_train.append(data['emotion'][i])
			#pixel_train.append(data['pixels'][i])
		elif 'PrivateTest' in  data['Usage'][i] :
			label_test.append(data['emotion'][i])
			#pixel_test.append(data['pixels'][i])
		else    :
			label_val.append(data['emotion'][i])
			#pixel_val.append(data['pixels'][i])
	#return label_train, pixel_train, label_test, pixel_test
	return label_train, label_test

def load_img(path, size):
	pixel_=[]
	for i in range(size):
		pixel_.append(cv2.imread(path+str(i+1)+'.jpg', 0))
	return pixel_

def load_all_data():
	smile_train_pixel, smile_train_label, smile_test_pixel, smile_test_label = load_data(SMILE_DATA)
	emotion_train_label, emotion_test_label = load_csv()
	emotion_train_pixel = load_img(EMOTION_IMAGE_TRAIN, 28709)
	emotion_test_pixel = load_img(EMOTION_IMAGE_TEST, 3589)

	return smile_train_pixel, smile_train_label, smile_test_pixel, smile_test_label, emotion_train_pixel, emotion_train_label, emotion_test_pixel, emotion_test_label

