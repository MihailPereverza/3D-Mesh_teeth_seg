import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import glob

if __name__ == '__main__':

    # data/data_part_1/upper/EKITH3BB
    data_path = os.getcwd() + '\\data'
    output_path = './'
    train_size = 0.8

    file_lists = glob.glob(data_path)
    # for file in file_lists[:300]:
    #     print(file)
    # data/data_part_1/upper/EKITH3BB
    # data/data_part_1/lower/O52P1SZT/O52P1SZT_lower.json

    sample_list = np.asarray(file_lists)
    print(sample_list)
    arr_tmp = []
    for sample in sample_list:
        sample_tmp = sample.split('\\')
        sample_ = sample + "\\" + sample_tmp[-1] + "_" + sample_tmp[-2]
        arr_tmp.append(sample_)

    train_list, val_list = train_test_split(arr_tmp, train_size=0.8, shuffle=True)
    val_list, test_list = train_test_split(val_list, train_size=0.5, shuffle=True)
    print('Training list:\n', train_list, '\nValidation list:\n', val_list, '\nTest list:\n', test_list)

    with open(os.path.join(output_path, 'train_list_{0}.csv'.format(1)), 'w') as file:
        for f in train_list:
            file.write(f+'\n')
    with open(os.path.join(output_path, 'val_list_{0}.csv'.format(1)), 'w') as file:
        for f in val_list:
            file.write(f+'\n')
    with open(os.path.join(output_path, 'test_list_{0}.csv'.format(1)), 'w') as file:
        for f in test_list:
            file.write(f+'\n')

    print('--------------------------------------------')
    print('# of train:', len(train_list))
    print('# of validation:', len(val_list))
    print('# of test:', len(test_list))
    print('--------------------------------------------')

