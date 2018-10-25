'''
    Save the data in the .csv file, save as a .npy file in ./data
'''

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
           'w', 'x', 'y', 'z']


def organize_data(folder):
    array = []
    labels = []

    for i in range(43):
        for j in range(26):
            for k in range(5):
                file_location = folder + "/student" + str(i) + "/" + letters[j] + "_" + str(k+1) + ".csv"
                file = open(file_location, 'r')
                this_file = []
                temp = file.readlines()
                for line in temp:
                    line = line.split(',')
                    line = line[1:]         # Get rid of first element since this is a label
                    for m in range(len(line)):
                        line[m].strip("\n")
                        line[m] = float(line[m])

                    this_file.append(line)

                array.append(np.array(this_file))
                labels.append(letters[j])
                file.close()

    label_encoder = LabelEncoder()
    oneh_encoder = OneHotEncoder()

    array = np.array(array)
    labels = np.array(labels)
    labels = label_encoder.fit_transform(labels).reshape(-1, 1)
    labels_new = oneh_encoder.fit_transform(labels).toarray()
    np.save('instances.npy', array)
    np.save('labels.npy', labels_new)

    return array, labels_new
