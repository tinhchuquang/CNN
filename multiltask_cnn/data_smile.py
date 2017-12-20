import numpy as np 
import cv2


OPENFILE = '/home/tinh/CNN/m_CNN/genki-r2009b/'
SAVEFILE = '/home/tinh/CNN/m_CNN/GENKI4K_GRAY/'

list_img = [10, 36, 126, 142, 207, 295, 353, 450, 464, 598, 601, 1003, 1021, 1074, 1166, 1213, 1245, 1316, 1352, 1378, 1405, 1939, 2405, 2435, 2673, 2715, 2716, 2845, 2893, 2982, 3283]
list_img_no = [104, 122, 299, 411, 466, 487, 498, 503, 624, 739, 1151, 1252, 1335, 1390, 1433, 1467, 1567, 1629, 1653, 1828, 1869, 1945, 2110, 2189, 2191, 2395, 2403, 2497, 2523, 2541, 2606, 2653, 2955, 3024, 3326, 3486, 3643, 3689, 3821, 3837, 3971]
label_img = [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1]
label_img_no = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1 , 1 ,0, 1, 1 ,1, 0, 1, 1, 0]


def replace_img(list_img):
	replace = np.arange(4000, 4031)
	for i in range(len(list_img)):
		img = cv2.imread(OPENFILE+str(replace[i])+'.jpg')
		cv2.imwrite(SAVEFILE+str(list_img[i])+'.jpg', img)

def insert_img(list_img):
	replace = np.arange(4950, 4991)
	for i in range(len(list_img)):
		img = cv2.imread(OPENFILE+str(replace[i])+'.jpg')
		cv2.imwrite(SAVEFILE+str(list_img[i])+'.jpg', img)

def load_label():
    SMILE_LABEL_FOLDER = '/home/tinh/CNN/m_CNN/GENKI4K/'
    X = []
    with open(SMILE_LABEL_FOLDER + "labels.txt") as f:
        for i in range(4000):
            l = f.readline()
            label = (int)(l.split()[0])
            X.append(label)
    for i in range(len(list_img)):
    	X[list_img[i]] = label_img[i]

    for i in range(len(list_img_no)):
    	X[list_img_no[i]] = label_img_no[i]
    file = open("/home/tinh/CNN/m_CNN/labels_smile.txt","w") 
    for i in range(len(X)):
    	file.write(str(X[i])+'\n')


load_label()
# replace_img(list_img)
# insert_img(list_img_no)