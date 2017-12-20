import numpy as np
import pickle
import cv2
import Image

SAVE_AGE = '/home/tinh/CNN/m_CNN/age_cnn/image_age/'

def load_gender():
	f=open('/home/tinh/CNN/faceAligmImdb/ageGender','rb')
	age=pickle.load(f)
	gender=pickle.load(f)
	f.close()
	# print(len(gender))
# for  i in range (len(age)):
#   print(age[i])
#   print(gender[i])

def move_img(begin ,size, path_source, path_des):
	for i in range(begin, begin+size):
		img = cv2.imread(path_source+str(i)+'.jpg')
		cv2.imwrite(path_des+str(i)+'.jpg', img)
def resize_img(begin ,size, path_source, path_des):
	for i in range(begin, begin+size):
		name_image = path_source + str(i) +'.jpg'
		img = Image.open(name_image)
		img = img.resize((48, 48), Image.ANTIALIAS)
		img.save(path_des+str(i)+'.jpg')

def move_img2(path_source, path_des, list_img):
	for i in list_img:
		img = cv2.imread(path_source+str(i+30000)+'.jpg')
		cv2.imwrite(path_des+str(i)+'.jpg', img)
def resize_img2(path_source, path_des, list_img):
	for i in list_img:
		name_image = path_source + str(i) +'.jpg'
		img = Image.open(name_image)
		img = img.resize((48, 48), Image.ANTIALIAS)
		img.save(path_des+str(i)+'.jpg')
def count_img_gray(path, size):
	index_of_list = []
	for i in range(size):
		img = cv2.imread(path+str(i)+'.jpg')
		if img.shape == np.zeros((48, 48, 3)).shape:
			index_of_list.append(i)

	return index_of_list


index_of_list = count_img_gray('/home/tinh/CNN/m_CNN/age_cnn/image_age/', 30000)
print(len(index_of_list))
# move_img(0, 30000, '/home/tinh/CNN/faceAligmImdb/','/home/tinh/CNN/m_CNN/age_cnn/image_age/')
# resize_img(0, 30000, '/home/tinh/CNN/m_CNN/age_cnn/image_age/','/home/tinh/CNN/m_CNN/age_cnn/image_age/')
# list_img = [61, 283, 297, 301, 315, 509, 884, 1337, 2726, 2881, 8276, 9429, 10000, 11346, 11503, 11962, 12258, 18181, 19253, 19598, 19851, 23592]
# move_img2('/home/tinh/CNN/faceAligmImdb/', '/home/tinh/CNN/m_CNN/age_cnn/image_age/', list_img)
# resize_img2('/home/tinh/CNN/m_CNN/age_cnn/image_age/', '/home/tinh/CNN/m_CNN/age_cnn/image_age/', list_img)
