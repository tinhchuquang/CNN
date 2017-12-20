import numpy as np 
import cv2
import scipy
import random


IMDB_FOLDER='./database/'
SMILE_FOLDER = '/database/'
GENDER_FOLDER = '/database/'
EMOTION_FOLDER = '/database/'

def getSmileImage():
    print('Load smile image.................')
    X1 = np.load(SMILE_FOLDER+'train_smile.npy')
    X2 = np.load(SMILE_FOLDER+'test_smile.npy')

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])
    print('Number of smile train data', str(len(test_data)))
    print('Done !')
    print('-----------------------------------------------------------------')
    return train_data, test_data

def getGenderImage():
    print('Load gender image....................')
    X1 = np.load(GENDER_FOLDER+'train_gender.npy')
    X2 = np.load(GENDER_FOLDER+'test_gender.npy')
    
    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Number of gender train data ', str(len(train_data)))
    print('Done')
    print('------------------------------------------------------------------')
    return train_data, test_data

def getAgeImage():
    print('Load age image................................')
    X1 = np.load(AGE_FOLDER+'train_age.npy')
    X2 = np.load(AGE_FOLDER+'test_age.npy')

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    return train_data, test_data



def getEmotionImage():
    print('Load emotion image............................')
    X1 = np.load(EMOTION_FOLDER+'train_emotion.npy')
    X2 = np.load(EMOTION_FOLDER+'test_emotion.npy')

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Number of gender train data ', str(len(train_data)))
    print('Done')
    print('------------------------------------------------------------------')
    return train_data, test_data

def getImdbAgeImage():
    print('Load age image...................')
    X1 = np.load(IMDB_FOLDER + 'train_age_imdb.npy')
    X2 = np.load(IMDB_FOLDER + 'test_age_imdb.npy')

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Number of age train data: ', str(len(train_data)))
    print('Done !')
    print('---------------------------------------------------------------')
    return train_data, test_data


def getImdbGenderImage():
    print('Load gender image................................................')
    X1 = np.load(IMDB_FOLDER+'train_gender_imdb.npy')
    X2 = np.load(IMDB_FOLDER+'test_gender_imdb.npy')

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X1.shape[0]):
        test_data.append(X2[i])

    print('Number of gender image...........................................')
    print('Done')
    print('-----------------------------------------------------------------')
    return train_data, test_data






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


def augmentation(batch, img_size):
    batch = random_crop(batch, (img_size, img_size), 10)
    #batch = random_blur(batch)
    batch = random_flip_leftright(batch)
    batch = random_rotation(batch, 10)

    return batch