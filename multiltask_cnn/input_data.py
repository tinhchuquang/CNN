import numpy as np
import os
import cv2
import scipy
import random
import imutils



SMILE_FOLDER = '/home/tinh/CNN/m_CNN/multiltask/database/'
EMOTION_FOLDER = '/home/tinh/CNN/m_CNN/multiltask/database/'
GENDER_FOLDER = '/home/tinh/CNN/m_CNN/multiltask/database/'
IMDB_FOLDER = '/home/tinh/CNN/m_CNN/multiltask/database/'
NUM_SMILE_IMAGE = 4000
SMILE_SIZE = 48
EMOTION_SIZE = 48


def getSmileImage():
    print('Load smile image...................')
    X1 = np.load(SMILE_FOLDER + 'train_smile.npy')
    X2 = np.load(SMILE_FOLDER + 'test_smile.npy')

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Done !')
    print('Number of smile train data: ',str(len(train_data)))
    print('---------------------------------------------------------------')
    return train_data, test_data


def getImdbImage():
    print('Load gender image...................')
    X1 = np.load(IMDB_FOLDER + 'train_imdb.npy')
    X2 = np.load(IMDB_FOLDER + 'test_imdb.npy')

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Done !')
    print('Number of gender train data: ', str(len(train_data)))
    print('---------------------------------------------------------------')
    return train_data, test_data

def getGenderImage():
    print('Load gender image...................')
    X1 = np.load(GENDER_FOLDER + 'train_gender.npy')
    X2 = np.load(GENDER_FOLDER + 'test_gender.npy')

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Done !')
    print('Number of gender train data: ', str(len(train_data)))
    print('---------------------------------------------------------------')
    return train_data, test_data


def getEmotionImage():
    print('Load emotion image..................')
    X1 = np.load(EMOTION_FOLDER+'train_emotion.npy')
    X2 = np.load(EMOTION_FOLDER+'public_test_emotion.npy')
    X3 = np.load(EMOTION_FOLDER+'private_test_emotion.npy')
    train_data = []
    public_test_data = []
    private_test_data = []

    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        public_test_data.append(X2[i])
    for i in range(X3.shape[0]):
        private_test_data.append(X3[i])    
    print('Done !')
    print('Number of emotion train data: ', str(len(train_data)))
    print('---------------------------------------------------------------')
    return train_data, public_test_data, private_test_data

def from_2d_to_3d(_batch_img):
    for i in range(len(_batch_img)):
        _batch_img[i] = _batch_img[i][:,:,np.newaxis]
    return _batch_img

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


def augmentation(batch, img_size):
    batch = random_crop(batch, (img_size, img_size), 10)
    #batch = random_blur(batch)
    batch = random_flip_leftright(batch)
    batch = random_rotation(batch, 10)

    return batch


def flip_data(img):
    # copy image to display all 4 variations
    vertical_img = img.copy()

    # flip img horizontally, vertically,
    # and both axes with flip()
    vertical_img = cv2.flip( img, 1 )
    return vertical_img
    
def rotate_image(image, angle):
    rotated = imutils.rotate(image, angle)
    return rotated


def crop_image(img, size):
    
    img = cv2.resize(img, (size+4, size+4))
    x = random.randint(0, img.shape[0] - size)
    y = random.randint(0, img.shape[0] - size)
    img = img[ y:y + size, x:x + size]

    return img

def shuffle_data(train_images, train_labels):
    arr = list(range(len(train_labels)))
    random.shuffle(arr)
    train_images = train_images[arr]
    train_labels = train_labels[arr]

    return train_images, train_labels

def augmentation_task(x_batch, i, size):
    random_mirror = random.randint(1, 2)
    angle = random.randint(-10, 10)
    img = x_batch[i]
    img = crop_image(img, size)
    # rotate
    img = rotate_image(img, angle)
    # mirror
    if random_mirror == 1:
        img = flip_data(img)  
    return img


def augmentation_batch(x_batch, size):
    # Parallel(n_jobs=4)(delayed(augmentation_task)(x_batch,i) for i in range(num_examples))
    num_examples = np.shape(x_batch)[0]
    new_img = []
    for i in range(num_examples):
        new_img.append(augmentation_task(x_batch, i, size))
    return new_img


def center_crop(x_batch):
    num_examples = np.shape(x_batch)[0]
    new_img = []
    for i in range(num_examples):
        img = x_batch[i]
        # center crop image to size (42,42)
        x = (48 - 42) // 2
        y = (48 - 42) // 2
        img = img[x:x + 42, y:y + 42]
        new_img.append(img)
    return new_img

# Create 5 crop images from original image
def five_crop(x_batch):
    x = (48 - 42) // 2
    y = (48 - 42) // 2
    batch_center = x_batch[:, x:x + 42, y:y + 42]
    batch_topleft = x_batch[:, :42:, :42:]
    batch_topright = x_batch[:, 6:48:, :42:]
    batch_downright = x_batch[:, 6:48:, 6:48:]
    batch_downleft = x_batch[:, :42:, 6:48:]

    return batch_center, batch_topleft, batch_topright, batch_downright, batch_downleft, \
           batch_center[:, :, ::-1], batch_topleft[:, :, ::-1], batch_topright[:, :, ::-1], batch_downright[:, :, ::-1], \
           batch_downleft[:, :, ::-1]