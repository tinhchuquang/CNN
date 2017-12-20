import numpy as np 
import pandas as pd 
import cv2
import pickle
import scipy
import random
import imutils


HOME='/home/tinh/CNN/fer2013/fer2013.csv'
TRAIN_EMOTION = '/home/tinh/CNN/fer2013/train/'
PUBLIC_TEST_EMOTION = '/home/tinh/CNN/fer2013/public_test/'
PRIVATE_TEST_EMOTON = '/home/tinh/CNN/fer2013/private_test/'
SMILE_PATH = '/home/tinh/CNN/m_CNN/SMILE_FOLDER/'
SAVE = '/home/tinh/CNN/m_CNN/multiltask/database/'
LABEL_TRAIN_GENDER = '/home/tinh/CNN/faceAligmImdb/ageGender'
TRAIN_GENDER = '/home/tinh/CNN/faceAligmImdb_train/'
LABEL_TEST_GENDER = '/home/tinh/CNN/faceAligmImdb/ageGender'
TEST_GENDER = '/home/tinh/CNN/faceAligmImdb_test/'


def load_label_gender(path,begin, size, list_img):
	label_gender = []
	f=open(path,'rb')
	age=pickle.load(f)
	gender=pickle.load(f)
	f.close()
	for i in range(begin, begin+size):
		if i in list_img:
			label_gender.append(gender[i+30000])
		else:
			label_gender.append(gender[i])
	return label_gender

def load_img_gender(path, begin, size):
	data = []
	for i in range(begin, begin+size):
		data.append(cv2.imread(path+str(i)+'.jpg', 0))
	return data


def load_csv(path):
	data = pd.read_csv(path)
	label_train = []
	label_test = []
	label_private_test= []
	for i in range(len(data)):
		if 'Training' in data['Usage'][i]:
			label_train.append(data['emotion'][i])
			#pixel_train.append(data['pixels'][i])
		elif 'PrivateTest' in  data['Usage'][i] :
			label_test.append(data['emotion'][i])
			#pixel_test.append(data['pixels'][i])
		else:
			label_private_test.append(data['emotion'][i])
			#pixel_val.append(data['pixels'][i])
	#return label_train, pixel_train, label_test, pixel_test
	return label_train, label_test, label_private_test

def load_img(path, size):
	pixel_=[]
	for i in range(size):
		pixel_.append(cv2.imread(path+str(i+1)+'.jpg', 0))
	return pixel_



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
	y_train = all_label[train,:]
	y_test = all_label[test, :]
	y_train = []
	y_test = []
	X_train = []
	X_test = []
	for i in range(4000):
		if i < 3000:
			X_train.append(all_image[index[i]])
			y_train.append(sum(all_label[index[i]]))
		else :
			X_test.append(all_image[index[i]])
			y_test.append(sum(all_label[index[i]]))

	return X_train, y_train, X_test, y_test



def covert_npy(pixel, label, name):
	data = []
	for i in range(len(pixel)):
		element = [pixel[i], label[i]]
		data.append(element)
	np.save(name, data)

def data_process(pixel, label):
	data = []
	for i in range(len(pixel)):
		element = [pixel[i], label[i]]
		data.append(element)

	return data

def getSmileImage():
	x_train_smile, label_train_smile, x_test_smile, label_test_smile = load_data(SMILE_PATH)
	train_data = data_process(x_train_smile, label_train_smile)
	test_data = data_process(x_test_smile, label_test_smile)
	return train_data, test_data


def getGenderImage():
	list_img = [61, 283, 297, 301, 315, 509, 884, 1337, 2726, 2881, 8276, 9429, 10000, 11346, 11503, 11962, 12258, 18181, 19253, 19598, 19851, 23592]
	label_test_gender = load_label_gender(LABEL_TEST_GENDER, 60000, 6455, list_img)
	label_train_gender = load_label_gender(LABEL_TRAIN_GENDER, 0, 27000, list_img)
	pixel_train_gender = load_img_gender(TRAIN_GENDER, 0, 27000)
	pixel_test_gender = load_img_gender(TEST_GENDER, 60000, 6455)
	train_data = data_process(pixel_train_gender, label_train_gender)
	test_data = data_process(pixel_test_gender, label_test_gender)
	return train_data, test_data


def getEmotionImage():
	label_train_emotion,label_private_test_emotion,label_public_test_emotion = load_csv(HOME)
	pixel_train_emotion = load_img(TRAIN_EMOTION,27000)
	pixel_public_test_emotion = load_img(PUBLIC_TEST_EMOTION, 3589)
	pixel_private_test_emotion = load_img(PRIVATE_TEST_EMOTON, 3589)
	train_data = data_process(pixel_public_test_emotion, label_public_test_emotion)
	public_test_data = data_process(pixel_train_emotion, label_train_emotion)
	private_test_data = data_process(pixel_private_test_emotion, label_private_test_emotion)
	return train_data, public_test_data, private_test_data


def from_2D_to_3D(_batch_img):
	for i in range(len(_batch_img)):
		_batch_img[i] = _batch_img[i][:,:,np.newaxis]
	return _batch_img

# pixel_train_gender = load_img_gender(TRAIN_GENDER, 0, 27000)
# print(pixel_train_gender[1])

label_train_emotion,label_private_test_emotion,label_public_test_emotion = load_csv(HOME)
pixel_train_emotion = load_img(TRAIN_EMOTION,len(label_train_emotion))
pixel_public_test_emotion = load_img(PUBLIC_TEST_EMOTION, 3589)
pixel_private_test_emotion = load_img(PRIVATE_TEST_EMOTON, 3589)
# x_train_smile, label_train_smile, x_test_smile, label_test_smile = load_data(SMILE_PATH)
# list_img = [61, 283, 297, 301, 315, 509, 884, 1337, 2726, 2881, 8276, 9429, 10000, 11346, 11503, 11962, 12258, 18181, 19253, 19598, 19851, 23592]
# label_test_gender = load_label_gender(LABEL_TEST_GENDER, 60000, 6455, list_img)
# label_train_gender = load_label_gender(LABEL_TRAIN_GENDER, 0, 27000, list_img)
# pixel_train_gender = load_img_gender(TRAIN_GENDER, 0, 27000)
# pixel_test_gender = load_img_gender(TEST_GENDER, 60000, 6455)
# pixel_train_gender = from_2D_to_3D(pixel_train_gender)
# pixel_test_gender = from_2D_to_3D(pixel_test_gender)
pixel_train_emotion = from_2D_to_3D(pixel_train_emotion)
pixel_public_test_emotion = from_2D_to_3D(pixel_public_test_emotion)
pixel_private_test_emotion = from_2D_to_3D(pixel_private_test_emotion)
# x_train_smile = from_2D_to_3D(x_train_smile)
# x_test_smile = from_2D_to_3D(x_test_smile)

covert_npy(pixel_public_test_emotion, label_public_test_emotion, SAVE + 'public_test_emotion.npy')
covert_npy(pixel_train_emotion, label_train_emotion, SAVE +'train_emotion.npy')
covert_npy(pixel_private_test_emotion, label_private_test_emotion, SAVE + 'private_test_emotion.npy')
# covert_npy(x_train_smile, label_train_smile, SAVE + 'train_smile.npy')
# covert_npy(x_test_smile, label_test_smile, SAVE + 'test_smile.npy')
# covert_npy(pixel_train_gender, label_train_gender, SAVE+'train_gender.npy')
# covert_npy(pixel_test_gender, label_test_gender, SAVE+'test_gender.npy')

# print(label_test_smile[0])

# print(pixel_train_gender[0].shape)

def flip_data(img):
	# copy image to display all 4 variations
	vertical_img = img.copy()

	# flip img horizontally, vertically,
	# and both axes with flip()
	vertical_img = cv2.flip( img, 1 )
	return vertical_img
	
def rotate_image(image, angle):
	rotated = imutils.rotate(image, angle)
	return rotated


def crop_image(img, size):
	
	img = cv2.resize(img, (size+4, size+4))
	x = random.randint(0, img.shape[0] - size)
	y = random.randint(0, img.shape[0] - size)
	print('x: ' + str(x))
	print('y: ' +str(y))
	img = img[ y:y + size, x:x + size]
	print('Img Size')
	print(img.shape)


	return img

def shuffle_data(train_images, train_labels):
	arr = list(range(len(train_labels)))
	random.shuffle(arr)
	train_images = train_images[arr]
	train_labels = train_labels[arr]

	return train_images, train_labels

def augmentation_task(x_batch, i, size):
	random_mirror = random.randint(1, 2)
	angle = random.randint(-10, 10)
	img = x_batch[i]
	img = crop_image(img, size)
	# rotate
	img = rotate_image(img, angle)
	# mirror
	if random_mirror == 1:
		img = flip_data(img)  
	return img


def augmentation_batch(x_batch, size):
	# Parallel(n_jobs=4)(delayed(augmentation_task)(x_batch,i) for i in range(num_examples))
	num_examples = np.shape(x_batch)[0]
	new_img = []
	for i in range(num_examples):
		new_img.append(augmentation_task(x_batch, i, size))
	return new_img


def center_crop(x_batch):
	num_examples = np.shape(x_batch)[0]
	new_img = []
	for i in range(num_examples):
		img = x_batch[i]
		# center crop image to size (42,42)
		x = (48 - 42) // 2
		y = (48 - 42) // 2
		img = img[x:x + 42, y:y + 42]
		new_img.append(img)
	return new_img

# Create 5 crop images from original image
def five_crop(x_batch):
	x = (48 - 42) // 2
	y = (48 - 42) // 2
	batch_center = x_batch[:, x:x + 42, y:y + 42]
	batch_topleft = x_batch[:, :42:, :42:]
	batch_topright = x_batch[:, 6:48:, :42:]
	batch_downright = x_batch[:, 6:48:, 6:48:]
	batch_downleft = x_batch[:, :42:, 6:48:]

	return batch_center, batch_topleft, batch_topright, batch_downright, batch_downleft, \
		   batch_center[:, :, ::-1], batch_topleft[:, :, ::-1], batch_topright[:, :, ::-1], batch_downright[:, :, ::-1], \
		   batch_downleft[:, :, ::-1]
