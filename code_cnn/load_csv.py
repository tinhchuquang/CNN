import pandas as pd 
import numpy as np

HOME='/home/tinh/CNN/fer2013/fer2013.csv'
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
		else 	:
			label_val.append(data['emotion'][i])
			pixel_val.append(data['pixels'][i])
	return label_train, pixel_train, label_test, pixel_test, label_val, pixel_val




label_train, pixel_train, label_test, pixel_test, label_val, pixel_test = load_csv(HOME)
print(len(label_val))

