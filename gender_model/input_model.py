import numpy as np
import scipy
import random
import cv2
from const import *
import pickle

GENDER_TRAIN_FOLDER = '/home/tinh/CNN/'
GENDER_TEST_FORLDER = '/home/tinh/CNN/'
LABEL_PATH = './save_data/data_gender_v2.pkl'


data = []
data_test = []
data_male = []
data_female = []


def global_image():
    global  data_male, data_female
    f = open(LABEL_PATH, 'rb')
    data_female = pickle.load(f)
    data_male = np.load(f)
    f.close()
    np.random.shuffle(data_male)
    np.random.shuffle(data_female)

def init():
    global data
    data = []

    for i in range(NUMBER_TRAIN_DATA/2):
        data.append(data_male[i])
    for i in range(NUMBER_TRAIN_DATA/2):
        data.append(data_female[i])
    np.random.shuffle(data)


def init_test():
    global data_test
    data_test = []

    for i in range(NUMBER_TRAIN_DATA/2, NUMBER_TRAIN_DATA/2 + 12800):
        data_test.append(data_male[i])
    for i in range(NUMBER_TRAIN_DATA / 2, NUMBER_TRAIN_DATA / 2 + 12800):
        data_test.append(data_female[i])

    np.random.shuffle(data_test)


def one_hot(index, num_classes):
    assert index < num_classes and index >= 0
    tmp = np.zeros(num_classes, dtype=np.float32)
    tmp[index] = 1.0
    return tmp


def get_train_image(point,batch_size):
    if point == 0:
        init()
    train_img = []
    i = 0
    while True:
        img = cv2.imread(data[i+point][1])
        train_img.append((img, one_hot((int)(data[i+point][0]), 2)))
        i += 1
        if len(train_img) >= batch_size:
            break

    point += i
    return train_img, point

def get_test_image(point,batch_size):
    if point == 0:
        init_test()
    test_img = []
    i = 0
    while True:
        img = cv2.imread(data_test[i+point][1])
        test_img.append((img, one_hot((int)(data_test[i+point][0]), 2)))
        i+=1
        if len(test_img) >= batch_size:
            break
    point += i

    return test_img, point


''' Data augmentation method '''


def random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])
    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad, mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    return new_batch


def random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def random_flip_updown(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.flipud(batch[i])
    return batch


def random_90degrees_rotation(batch, rotations=[0, 1, 2, 3]):
    for i in range(len(batch)):
        num_rotations = random.choice(rotations)
        batch[i] = np.rot90(batch[i], num_rotations)
    return batch


def random_rotation(batch, max_angle):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            angle = random.uniform(-max_angle, max_angle)
            batch[i] = scipy.ndimage.interpolation.rotate(batch[i], angle, reshape=False)
    return batch


def random_blur(batch, sigma_max=5.0):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            sigma = random.uniform(0., sigma_max)
            batch[i] = scipy.ndimage.filters.gaussian_filter(batch[i], sigma)
    return batch

def augmentation(batch, (width_size, height_size)):
    #batch = random_crop(batch, (with_szie, height_size), 10)
    #batch = random_blur(batch)
    batch = random_flip_leftright(batch)
    #batch = random_rotation(batch, 10)

    return batch
