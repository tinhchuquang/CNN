import pandas as pd
import cv2
import numpy as np


HOME = '/home/tinh/CNN/fer2013/fer2013.csv'
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
			pixel_train.append(data['pixels'][i])
		elif 'PrivateTest' in  data['Usage'][i] :
			label_test.append(data['emotion'][i])
			pixel_test.append(data['pixels'][i])
		else    :
			label_val.append(data['emotion'][i])
			pixel_val.append(data['pixels'][i])
	return label_train, pixel_train, label_test, pixel_test, label_val, pixel_val

label_train,pixel_train,label_test, pixel_test, label_val, pixel_val = load_csv(HOME)

# for i in range(len(pixel_train)):
# 	pixel_train[i] = np.matrix(pixel_train[i])
# 	pixel_train[i] = pixel_train[i].reshape(48, 48)
# 	#pixel_train[i] = pixel_train[i][:,:,np.newaxis]
# 	cv2.imwrite('/home/tinh/CNN/fer2013/train/'+str(i+1)+'.jpg', pixel_train[i])
# for i in range(len(pixel_test)):
# 	pixel_test[i] = np.matrix(pixel_test[i])
# 	pixel_test[i] = pixel_test[i].reshape(48, 48)
# 		#pixel_test[i] = pixel_test[i][:, :, np.newaxis]
# 	cv2.imwrite('/home/tinh/CNN/fer2013/test/'+str(i+1)+'.jpg', pixel_test[i])

print(len(pixel_val))
for i in range(len(pixel_val)):
	pixel_val[i] = np.matrix(pixel_val[i])
	pixel_val[i] = pixel_val[i].reshape(48, 48)
		#pixel_test[i] = pixel_test[i][:, :, np.newaxis]
	cv2.imwrite('/home/tinh/CNN/fer2013/private_test/'+str(i+1)+'.jpg', pixel_val[i])
