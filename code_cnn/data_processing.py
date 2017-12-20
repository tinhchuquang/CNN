import numpy as np
import random
import imutils
import cv2

def normalize_data(data, X_mean = None, X_std = None):
    num_row=np.shape(data)[0]
    cc=150
    data=data/255.
    mean=np.mean(data,axis=0).reshape((1,2304))
    sum=np.sum(data,axis=1)
    data[sum<1e-2,:]=mean

    #normalize per image
    data=data-np.mean(data,1).reshape((num_row,1))
    data_norm=np.sqrt(np.sum(data**2,1))
    data_norm[data_norm<1e-8]=1
    data_norm=np.reshape(data_norm,(num_row,1))
    data=cc*data/data_norm

    #normalize per pixel
    if X_mean is None:
        X_mean= mean
        X_std=np.std(data,axis=0).reshape((1,2304))

    data=(data-X_mean)/X_std

    return data.reshape((-1, 48, 48)), X_mean, X_std

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
     #crop image to size (42,42)
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