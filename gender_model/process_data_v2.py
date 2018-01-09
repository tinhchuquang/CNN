import numpy as np
import pickle
import cv2

prob_gender = [0, 0, 0]
data_female = []
data_male = []

NEW_PATH = '/home/tinh/CNN/m_CNN/age_model/'
PRE_NAME = 'coarse_tilt_aligned_face.'

GENDER_FOLDER = '/home/tinh/CNN/'
LABEL_PATH = '/home/tinh/CNN/imdb_aligm/meta_data.pkl'

f = open(LABEL_PATH, 'rb')
age = pickle.load(f)
gender = pickle.load(f)
path =  pickle.load(f)

f.close()



if __name__ == "__main__":

    for i in range(len(gender)):
        if np.isnan(gender[i]):
            prob_gender[2] += 1
        else:
            if gender[i] == 1:
                data_male.append(((int(gender[i])), GENDER_FOLDER+path[i][3:]))
                print(GENDER_FOLDER+path[i][3:])
                prob_gender[1] += 1
            else:
                data_female.append((int(gender[i]), GENDER_FOLDER+path[i][3:]))
                prob_gender[0] += 1

    for i in range(5):
        name = NEW_PATH + 'fold_' + str(i) + '_data.txt'
        file = open(name, 'r')
        j = 0
        for line in file:
            if j != 0:
                # A = line.split()
                for k in range(len(line)):
                    if line[k] == 'f':
                        A = line.split()
                        path = NEW_PATH+'faces_align/'+ (A[0])+'/'+PRE_NAME+A[2]+'.' +(A[1])
                        print(path)
                        img = cv2.imread(path)
                        if img is not None:
                            data_female.append((0, path))
                        break
                    if line[k] == 'm':
                        A = line.split()
                        path = NEW_PATH +'faces_align/'+ (A[0]) + '/' + PRE_NAME + A[2] + '.' + (A[1])
                        img = cv2.imread(path)
                        if img is not None:
                            data_male.append((1, path))
                        break
            else:
                j += 1

    f = open('./save_data/data_gender_v2.pkl', 'wb')
    pickle.dump(data_female, f)
    pickle.dump(data_male, f)
    f.close()
    # np.save('./save_data/data_female.npy', data_female)
    # np.save('./save_data/data_male.npy', data_male)

    print(len(data_male))
    print(len(data_female))