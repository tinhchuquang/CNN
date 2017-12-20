import cv2
import pickle 
import numpy as np 


def one_hot(index, num_classes):
    assert index < num_classes and index >= 0
    tmp = np.zeros(num_classes, dtype=np.float32)
    tmp[index] = 1.0
    return tmp

def PrepareIMDB():
    IMDB_FOLDER = '/home/tinh/CNN/faceAligmImdb/'

    F_FOLDER = '/home/tinh/CNN/m_CNN/age_cnn/'

    f = open('/home/tinh/CNN/faceAligmImdb/ageGender', 'rb')
    age = pickle.load(f)
    gender = pickle.load(f)
    f.close()
    n = len(age)

    X = []

    for i in range(n):
        fileName = IMDB_FOLDER + str(i) + '.jpg'
        img = cv2.imread(fileName)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(48, 48))

        T = np.zeros([48, 48, 1])
        T[:, :, 0] = img

        label = age[i]

        if label < 6:
        	label = 0
        elif label < 11:
        	label = 1
        elif label < 15:
        	label = 2
        elif label < 18:
        	label = 3
        elif label < 23:
        	label = 4
        elif label < 30:
        	label = 5
        elif label < 40:
        	label = 6
        elif label < 50:
        	label = 7
        elif label < 60:
        	label = 8
        elif label < 70:
        	label = 9
        elif label < 80:
        	label = 10
        elif label < 90:
        	label = 11
        else:
        	label = 12
        # print(label)
        X.append((T, one_hot(label, 13)))

    for _ in range(10):
        np.random.shuffle(X)

    train_data, test_data = X[:150000], X[150000:]

    np.save(F_FOLDER + 'train_imdb.npy', train_data)
    np.save(F_FOLDER + 'data_imdb.npy', X)
    np.save(F_FOLDER + 'test_imdb.npy', test_data)

PrepareIMDB()