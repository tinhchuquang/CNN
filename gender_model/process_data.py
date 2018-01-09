import numpy as np
import scipy
import random
import cv2
import pickle

AGE_TRAIN_FOLDER = '/home/tinh/CNN/'
AGE_TEST_FORLDER = '/home/tinh/CNN/'
LABEL_PATH = '/home/tinh/CNN/imdb_aligm/meta_data.pkl'

f = open(LABEL_PATH, 'rb')
age = pickle.load(f)
gender = pickle.load(f)
path =  pickle.load(f)

f.close()

prob_gender = [0, 0, 0]
data_female = []
data_male = []

if __name__ == "__main__":
    for i in range(len(gender)):
        if np.isnan(gender[i]):
            prob_gender[2] += 1
        else:
            if gender[i] == 1:
                data_male.append(((int(gender[i])), path[i]))

                prob_gender[1] += 1
            else:
                data_female.append((int(gender[i]), path[i]))
                prob_gender[0] += 1

    f = open('./save_data/data_gender.pkl', 'wb')
    pickle.dump(data_female, f)
    pickle.dump(data_male, f)
    f.close()
    # np.save('./save_data/data_female.npy', data_female)
    # np.save('./save_data/data_male.npy', data_male)
    print(prob_gender)