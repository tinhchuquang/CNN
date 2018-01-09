import cv2

NEW_PATH = './../age_model/'
PRE_NAME = 'coarse_tilt_aligned_face.'

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
                    path = NEW_PATH + 'faces_align/' + (A[0]) + '/' + PRE_NAME + A[2] + '.' + (A[1])
                    img = cv2.imread(path)
                    if img is None:
                        print("Image don't have")
                    break
                if line[k] == 'm':
                    A = line.split()
                    path = NEW_PATH + 'faces_align/' + (A[0]) + '/' + PRE_NAME + A[2] + '.' + (A[1])
                    img = cv2.imread(path)
                    if img is None:
                        print("Image don't have")
                    break
        else:
            j += 1