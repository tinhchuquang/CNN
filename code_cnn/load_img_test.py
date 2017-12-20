
import Image
import numpy as np


def load_image( infilename ):
    img = Image.open(infilename)
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

SAVE_FOLDER = '/home/tinh/CNN/m_CNN/GENKI4K/files/'
NEW_FOLDER = '/home/tinh/CNN/m_CNN/GENKI4K_CROP/'
SMALL_FOLDER = '/home/tinh/CNN/m_CNN/GENKI4K_SMALL/'
SMILE_FOLDER = '/home/tinh/CNN/m_CNN/SMILE_FOLDER/'

def gray_data():
	for i in range(4000):
		name_image = NEW_FOLDER +'file'
		if i < 9:
			name_image = name_image + '000'+ str(i+1)+'.jpg'
			img = Image.open(name_image).convert('L')
			img.save('/home/tinh/CNN/m_CNN/GENKI4K_GRAY/file'+ '000'+str(i+1)+ '.jpg')
		elif i < 99:
			name_image = name_image + '00'+ str(i+1)+'.jpg'
			img = Image.open(name_image).convert('L')
			img.save('/home/tinh/CNN/m_CNN/GENKI4K_GRAY/file'+ '00'+str(i+1)+ '.jpg')
		elif i < 999:
			name_image = name_image + '0'+ str(i+1)+'.jpg'
			img = Image.open(name_image).convert('L')
			img.save('/home/tinh/CNN/m_CNN/GENKI4K_GRAY/file'+ '0'+str(i+1)+ '.jpg')
		else:
			name_image = name_image + str(i+1)+'.jpg'
			img = Image.open(name_image).convert('L')
			img.save('/home/tinh/CNN/m_CNN/GENKI4K_GRAY/file'+str(i+1)+ '.jpg')

def resize_data():
    for i in range(4000):
        name_image = NEW_FOLDER + 'file'
        if i < 9:
        	name_image = name_image + '000'+ str(i+1) +'.jpg'
        	img = Image.open(name_image)
        	img = img.resize((100, 100), Image.ANTIALIAS)
        	img.save(SMALL_FOLDER+ 'file'+'000'+str(i+1)+'.jpg')
        elif i < 99:
            name_image = name_image + '00'+ str(i+1) +'.jpg'
            img = Image.open(name_image)
            img = img.resize((100, 100), Image.ANTIALIAS)
            img.save(SMALL_FOLDER+'file'+'00'+str(i+1)+'.jpg')
        elif i < 999:
            name_image = name_image + '0'+ str(i+1) +'.jpg'
            img = Image.open(name_image)
            img = img.resize((100, 100), Image.ANTIALIAS)
            img.save(SMALL_FOLDER+'file'+'0'+str(i+1)+'.jpg')
        else:
            name_image = name_image + str(i+1) +'.jpg'
            img = Image.open(name_image)
            img = img.resize((100, 100), Image.ANTIALIAS)
            img.save(SMALL_FOLDER+'file'+str(i+1)+'.jpg')

def resize_data2():
    for i in range(4000):
        name_image = SMALL_FOLDER
        if i < 9:
            name_image = name_image + str(i+1) +'.jpg'
            img = Image.open(name_image)
            img = img.resize((48, 48), Image.ANTIALIAS)
            img.save(SMILE_FOLDER+ str(i+1)+'.jpg')
        elif i < 99:
            name_image = name_image + str(i+1) +'.jpg'
            img = Image.open(name_image)
            img = img.resize((48, 48), Image.ANTIALIAS)
            img.save(SMILE_FOLDER+str(i+1)+'.jpg')
        elif i < 999:
            name_image = name_image + str(i+1) +'.jpg'
            img = Image.open(name_image)
            img = img.resize((48, 48), Image.ANTIALIAS)
            img.save(SMILE_FOLDER+str(i+1)+'.jpg')
        else:
            name_image = name_image + str(i+1) +'.jpg'
            img = Image.open(name_image)
            img = img.resize((48, 48), Image.ANTIALIAS)
            img.save(SMILE_FOLDER+str(i+1)+'.jpg')


def load_data():
    all_image = []
    all_label = np.zeros((4000, 1))
    
    for i in range(4000):
        name_image = NEW_FOLDER + 'file'
        if i < 9:
            name_image = name_image + '000' + str(i+1) +'.jpg'
            all_image.append(load_image(name_image))
        elif i < 99:
            name_image = name_image + '00' + str(i+1) +'.jpg'
            all_image.append(load_image(name_image))
        elif i < 999:
            name_image = name_image + '0'+ str(i+1) + '.jpg'
            all_image.append(load_image(name_image))
        else:
            name_image = name_image +str(i+1) +'.jpg'
            all_image.append(load_image(name_image))
    labels = open('/home/tinh/CNN/m_CNN/GENKI4K/labels.txt', 'r')
    index = 0
    for line in labels:
        all_label[index] = (line[0])
        index+= 1
    #shuffle data

    for i in range(len(all_image)):
	if (all_image[i].shape) == np.zeros((200, 200)).shape:
		print(i)
		print(all_image[i].shape)

    all_label = np.asarray(all_label, dtype='int32')



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

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)


def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)


def covert_2():
    img = Image.open(NEW_FOLDER+'file0001.jpg')
    arr = np.load('/home/tinh/CNN/SmileDataset/test.npy')
    img2 = array2PIL(arr[0][0], 100)
    img2.save('out.jpg')



#covert_2()
# name_image = '/home/tinh/CNN/m_CNN/GENKI-R2009a/files/file0000000000004784.jpg'
# img = Image.open(name_image)
# img = img.resize((200, 200), Image.ANTIALIAS)
# img.save('/home/tinh/CNN/m_CNN/GENKI4K_CROP/file'+'3975'+'.jpg')
# train_img, train_label, test_img, test_label = load_data()
#resize_data()
#gray_data()

# a = load_image('/home/tinh/CNN/m_CNN/GENKI4K_CROP/file0001.jpg')
# print(a.shape)

# for i in range(len(train_img)):
# 	if (train_img[i].shape) == np.zeros((200, 200)).shape:
# 		print(i)
# 		print(train_img[i].shape)
resize_data2()



# batch_size = 128
# num_batch = int(len(train_img) // batch_size)

# for batch in range(num_batch):
# 	top = batch * batch_size
# 	bot = min((batch + 1) * batch_size, len(train_img))
# 	print(train_img[top: bot])
# 	batch_img = np.array(train_img[top:bot])
# 	batch_label = np.array(train_label[top:bot])
# 	print(batch_img.shape)




#print(data.shape)
#print(data[0:, 0, 0])

# text = open('/home/tinh/CNN/m_CNN/GENKI4K/labels.txt', 'r')

# a = 3 // 2 
# print(a)