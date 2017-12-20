import pickle
import cv2
import numpy as np


def PrepareSmileData():
    SMILE_LABEL_FOLDER = '/home/tinh/CNN/m_CNN/GENKI4K/'
    SMILE_FOLDER = '/home/tinh/CNN/m_CNN/SMILE_FOLDER/'
    F_SMILE_FOLDER = '/home/tinh/CNN/m_CNN/multiltask/database/'
    NUM_SMILE_IMAGE = 4000
    SMILE_SIZE = 48

    X = []
    with open(SMILE_LABEL_FOLDER + "labels.txt") as f:
        for i in range(NUM_SMILE_IMAGE):
            fileName = SMILE_FOLDER + str(i+1) + ".jpg"
            img = cv2.imread(fileName, 0)
            T = np.zeros([SMILE_SIZE, SMILE_SIZE, 1])
            T[:, :, 0] = img
            l = f.readline()
            label = (int)(l.split()[0])
            X.append((T, label))
    for _ in range(10):
        np.random.shuffle(X)

    train_data, test_data = X[:3000], X[3000:]

    np.save(F_SMILE_FOLDER + 'train_smile.npy', train_data)
    np.save(F_SMILE_FOLDER + 'test_smile.npy', test_data)

def PrepareGenderData():
    GENDER_FOLDER = '/home/tinh/CNN/m_CNN/multiltask/database/'

    full_data = np.load(GENDER_FOLDER + 'data_imdb.npy')
    # for i in range(len(full_data)):
        # full_data[i][0] = cv2.cvtColor(full_data[i][0], cv2.COLOR_BGR2GRAY)
        # full_data[i][0] = cv2.resize(full_data[i][0], (48, 48))
        # T = np.zeros([48, 48, 1])
        # T[:, :, 0] = full_data[i][0]
        # full_data[i][0] = T
        # img = full_data[i][0]
        # cv2.imshow('img', img/255.0)
        # print(full_data[i][1])
        # cv2.waitKey(0)

    n = len(full_data)
    train_data = []
    test_data = []

    for i in range(5):
        np.random.shuffle(full_data)

    for i in range(30000):
        train_data.append(full_data[i])
    for i in range(30000, n):
        test_data.append(full_data[i])

    np.save(GENDER_FOLDER + 'train_gender.npy', train_data)
    np.save(GENDER_FOLDER + 'test_gender.npy', test_data)

def PrepareIMDB():
    IMDB_FOLDER = '/home/tinh/CNN/faceAligmImdb/'

    F_FOLDER = '/home/tinh/CNN/m_CNN/multiltask/database/'

    f = open('/home/tinh/CNN/faceAligmImdb/ageGender', 'rb')
    age = pickle.load(f)
    gender = pickle.load(f)
    f.close()
    n = len(gender)

    X = []

    for i in range(n):
        fileName = IMDB_FOLDER + str(i) + '.jpg'
        img = cv2.imread(fileName, 0)
        img = cv2.resize(img,(48, 48))

        T = np.zeros([48, 48, 1])
        T[:, :, 0] = img

        label = gender[i]
        X.append((T, label))

    for _ in range(10):
        np.random.shuffle(X)

    train_data, test_data = X[:150000], X[150000:]

    np.save(F_FOLDER + 'train_imdb.npy', train_data)
    np.save(F_FOLDER + 'data_imdb.npy', X)
    np.save(F_FOLDER + 'test_imdb.npy', test_data)


# PrepareIMDB()
PrepareSmileData()
# PrepareGenderData()