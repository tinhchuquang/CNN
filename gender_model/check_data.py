from const import *
import numpy as np
import input_model as input_data

if __name__ == "__main__":
    num_batch = NUMBER_TRAIN_DATA // BATCH_SIZE
    point = 0
    input_data.global_image()
    for i in range(num_batch):
        data, point = input_data.get_train_image(point, BATCH_SIZE)
        print('Batch:' + str(num_batch) + ' lenght data:' + str(len(data)) + ' point:' + str(point))

    num_batch = NUMBER_TEST_DATA // BATCH_SIZE
    point = 0
    for i in range(num_batch):
        data, point = input_data.get_test_image(point, BATCH_SIZE)
        print('Batch:' + str(num_batch) + ' lenght data:' + str(len(data)) + ' point:' + str(point))