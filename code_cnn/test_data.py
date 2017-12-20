from data_processing import five_crop, shuffle_data, augmentation_batch
import pandas as pd
import numpy as np 
import cv2
import Image
import imutils

def load_image( infilename ) :
	img = Image.open( infilename )
	img.load()
	data = np.asarray( img, dtype="int32" )
	return data

HOME = '/home/tinh/CNN/fer2013/fer2013small.csv'
def load_data(SAVE_FOLDER):
	all_image = []
	all_label = np.zeros((4000, 1))
	
	for i in range(4000):
		name_image = SAVE_FOLDER
		if i < 9:
			name_image = name_image + str(i+1) +'.jpg'
			all_image.append(load_image(name_image))
		elif i < 99:
			name_image = name_image +str(i+1) +'.jpg'
			all_image.append(load_image(name_image))
		elif i < 999:
			name_image = name_image + str(i+1) + '.jpg'
			all_image.append(load_image(name_image))
		else:
			name_image = name_image +str(i+1) +'.jpg'
			all_image.append(load_image(name_image))
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
	# y_train = all_label[train,:]
	# y_test = all_label[test, :]
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

def load_data2(SAVE_FOLDER):
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
	# y_train = all_label[train,:]
	# y_test = all_label[test, :]
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

def convert_to_matrix(pixel_train=None, pixel_test=None):
	if pixel_train != None:
		for i in range(len(pixel_train)):
			pixel_train[i] = np.matrix(pixel_train[i])
			pixel_train[i] = pixel_train[i].reshape(48, 48)
			pixel_train[i] = pixel_train[i][:,:,np.newaxis]
	if pixel_test != None:
		for i in range(len(pixel_test)):
			pixel_test[i] = np.matrix(pixel_test[i])
			pixel_test[i] = pixel_test[i].reshape(48, 48)
			pixel_test[i] = pixel_test[i][:, :, np.newaxis]
	return pixel_train, pixel_test

def rotate_image(image, angle):
	rotated = imutils.rotate_bound(image, angle)
	cv2.imshow("Rotated (Correct)", rotated)
	cv2.waitKey(0)
	return rotate

def from_2D_to_3D(_batch_img):
	for i in range(len(_batch_img)):
		_batch_img[i] = _batch_img[i][:,:,np.newaxis]
	return _batch_img

def flip_data(img):
	# load the image with imread()
 
# copy image to display all 4 variations
	horizontal_img = img.copy()
	vertical_img = img.copy()
	both_img = img.copy()
 
# flip img horizontally, vertically,
# and both axes with flip()
	horizontal_img = cv2.flip( img, 0 )
	vertical_img = cv2.flip( img, 1 )
	both_img = cv2.flip( img, -1 )
 
# display the images on screen with imshow()
	cv2.imshow( "Original", img )

	cv2.imshow( "Vertical flip", vertical_img )

 
 
# wait time in milliseconds
# this is required to show the image
# 0 = wait indefinitely
	cv2.waitKey(0)
 
# close the windows
	cv2.destroyAllWindows()


def load_csv(path):
	data = pd.read_csv(path)
	label_train = []
	pixel_train = []
	label_test = []
	pixel_test = []
	label_val= []
	pixel_val = []
	for i in range(len(data)):
		pixel_train.append(data['pixels'][i])
	return label_train, pixel_train, label_test, pixel_test



label_train,pixel_train,label_test, pixel_test = load_csv(HOME)
for i in range(len(pixel_train)):
	pixel_train[i] = np.matrix(pixel_train[i])
	pixel_train[i] = pixel_train[i].reshape(48, 48)
	#pixel_train[i] = pixel_train[i][:,:,np.newaxis]
for i in range(len(pixel_test)):
	pixel_test[i] = np.matrix(pixel_test[i])
	pixel_test[i] = pixel_test[i].reshape(48, 48)
	#pixel_test[i] = pixel_test[i][:, :, np.newaxis]
for epoch in range(100):
	index = np.arange(5)
	np.random.shuffle(index)
	train_img = []
	train_label = []
	for i in index:
		train_img.append(pixel_train[i])


		# np.random.shuffle(smile_train)
		# train_img = []
		# train_label = []
		# for i in range(len(smile_train)):
		#     train_img.append(smile_train[i][0])
		#     train_label.append(one_hot(smile_train[i][1], 2))
	train_img = augmentation_batch(train_img)
	train_img = from_2D_to_3D(train_img)


# print(train_pixel[1].shape)
# for i in range(3000):
#   train_pixel[i] = train_pixel[i][:,:,np.newaxis]
# for i in range(1000):
#   test_pixel[i] = test_pixel[i][:, :, np.newaxis]

# index = np.arange(3000)
# np.random.shuffle(index)
# train_img = []
# train_label = []
# for i in index:
#   train_img.append(train_pixel[i])

	# np.random.shuffle(smile_train)
	# train_img = []
	# train_label = []
	# for i in range(len(smile_train)):
	#     train_img.append(smile_train[i][0])
	#     train_label.append(one_hot(smile_train[i][1], 2)

#flip_data(train_img[0])
# batch_img = np.asarray(train_img[0:5])
# print(batch_img[0].shape)
# for i in range(5):
#   cv2.imshow(str(i)+'origin', batch_img[i])
#   cv2.waitKey(0)
# batch_img = augmentation_batch(batch_img)
# print(batch_img[0].shape)
# for i in range(5):
#   cv2.imshow(str(i)+'augmentation', batch_img[i])
#   cv2.waitKey(0)

# pixel_train,_,_,_ = load_data(HOME)
# #pixel_train = convert_to_matrix(pixel_train)
# X = pixel_train[1]
# cv2.imwrite('origin_data.jpg', X)
# Y = augmentation_batch(X)
# cv2.imwrite('flip_data.jpg', Y)

# index = np.arange(1000)
# np.random.shuffle(index)
# test_img = []
# test_label = []
# for i in index:
#   test_img.append(test_pixel[i])
# print(test_img[i].shape)
# test_img = from_2D_to_3D(test_img)