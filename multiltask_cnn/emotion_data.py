import numpy as np
import pandas as pd
import os

SRC_PATH = '/home/tinh/CNN/fer2013/fer2013.csv'
DST_PATH = './database/'
data = pd.read_csv(SRC_PATH)

train_data = []
private_test_data = []
public_test_data = []

num_samples = np.zeros([7])

def make_one_hot(label):
    res = np.zeros(7)
    res[label] = 1.0
    return res


def data_to_image(pixels):
    data_image = np.fromstring(str(pixels), dtype=np.uint8, sep=' ')
    return data_image


for index, row in data.iterrows():
    pix = np.reshape(data_to_image(row['pixels']), (48, 48, 1)) / 255.0
    num_samples[int(row['emotion'])] += 1
    emotion = make_one_hot(int(row['emotion']))
    usage = row['Usage']
    if row['Usage'] == 'Training':
        train_data.append((pix, emotion))
    else:
        if row['Usage'] == 'PublicTest':
            public_test_data.append((pix, emotion))
        else:
            private_test_data.append((pix, emotion))

print(num_samples)
if not os.path.isdir(DST_PATH):
    os.mkdir(DST_PATH)
np.save(DST_PATH + 'train_emotion.npy', train_data)
np.save(DST_PATH + 'public_test_emotion.npy', public_test_data)
np.save(DST_PATH + 'private_test_emotion.npy', private_test_data)
