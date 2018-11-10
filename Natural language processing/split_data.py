"""
    3.1 Create train/validation/test splits

    This script will split the data/data.tsv into train/validation/test.tsv files.
"""
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_data():
    feats1 = []
    labels1 = []
    feats0 = []
    labels0 = []

    file = open("data/data.tsv", 'r')
    temp = file.readlines()

    for i in range(len(temp)):
        line = temp[i].split('\t')
        line[1].strip('\n')
        if i > 0:
            if int(line[1]) == 1:
                feats1.append(line[0])
                labels1.append(int(line[1]))
            if int(line[1]) == 0:
                feats0.append(line[0])
                labels0.append(int(line[1]))

    feats1 = np.array(feats1)
    labels1 = np.array(labels1)
    feats0 = np.array(feats0)
    labels0 = np.array(labels0)

    # Training: 64%, validation: 16%, test: 20%
    temp_features1, test_features1, temp_labels1, test_labels1 = \
        train_test_split(feats1, labels1, test_size=0.2, random_state=0)
    train_features1, valid_features1, train_labels1, valid_labels1 = \
        train_test_split(temp_features1, temp_labels1, test_size=0.2, random_state=0)

    temp_features0, test_features0, temp_labels0, test_labels0 = \
        train_test_split(feats0, labels0, test_size=0.2, random_state=0)
    train_features0, valid_features0, train_labels0, valid_labels0 = \
        train_test_split(temp_features0, temp_labels0, test_size=0.2, random_state=0)

    train_features = np.concatenate((train_features0, train_features1))

    valid_features = np.concatenate((valid_features0, valid_features1))
    test_features = np.concatenate((test_features0, test_features1))
    train_labels = np.concatenate((train_labels0, train_labels1))
    valid_labels = np.concatenate((valid_labels0, valid_labels1))
    test_labels = np.concatenate((test_labels0, test_labels1))

    train_data = np.stack((train_features, train_labels), axis=1)
    valid_data = np.stack((valid_features, valid_labels), axis=1)

    df1 = pd.DataFrame({"text": train_features, "label": train_labels})
    df1.to_csv('data/train.tsv', sep='\t')
    df2 = pd.DataFrame({"text": valid_features, "label": valid_labels})
    df2.to_csv('data/validation.tsv', sep='\t')
    df3 = pd.DataFrame({"text": test_features, "label": test_labels})
    df3.to_csv('data/test.tsv', sep='\t')

    test_data = np.stack((test_features, test_labels), axis=1)
    np.savetxt("train.tsv", train_data, delimiter='\t', fmt='%s')
    np.savetxt("validation.tsv", valid_data, delimiter='\t', fmt='%s')
    np.savetxt("test.tsv", test_data, delimiter='\t', fmt='%s')

    return


if __name__ == "__main__":
    split_data()
