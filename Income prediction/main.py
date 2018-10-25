import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MultiLayerPerceptron
from dataset import AdultDataset
from util import *

import matplotlib.pyplot as plt


""" Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""

# Hyper-parameters
seed1 = 0
seed2 = 42
learning_rate = 0.1     # 0.001, 0.01, 0.1, 1, 10, 100, 1000 (best = 0.1)
batch_size = 64         # 1 (2), 64, 17932 (best = 64)
num_epochs = 20         # Default = 20
eval_every = 10         # Default = 10


# =================================== LOAD DATASET =========================================== #

######

# 2.1 YOUR CODE HERE

data = pd.read_csv("adult.csv")

######

# =================================== DATA VISUALIZATION =========================================== #

# the dataset is imported as a DataFrame object, for more information refer to
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# we can check the number of rows and columns in the dataset using the .shape field
# to get a taste of what our datset looks like, let's visualize the first 5 rows of the dataset using the .head() method
# the task will be to predict the "income" field (>50k or <50k) based on the other fields in the dataset
# check how balanced our dataset is using the .value_counts() method.

######

# 2.2 YOUR CODE HERE

# print(data.shape)
# print(data.columns)
# verbose_print(data.head())
# print(data["income"].value_counts())

######


# =================================== DATA CLEANING =========================================== #

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

# let's first count how many missing entries there are for each feature
col_names = data.columns
num_rows = data.shape[0]

######

# 2.3 YOUR CODE HERE

# print(data.isin(["?"]).sum())

######

# next let's throw out all rows (samples) with 1 or more "?"
# Hint: take a look at what data[data["income"] != ">50K"] returns
# Hint: if data[field] do not contain strings then data["income"] != ">50K" will return an error


for feature in col_names:
    ######

    # 2.3 YOUR CODE HERE
    total_missing = data[feature].isin(["?"]).sum()

    if total_missing > 0:
        data = data[data[feature] != "?"]

# print(data.shape)
# print("Rows removed: ", num_rows - data.shape[0])

######

# =================================== BALANCE DATASET =========================================== #

######

# 2.4 YOUR CODE HERE

more_than_50k = data["income"].isin([">50K"]).sum()
less_than_50k = data["income"].isin(["<=50K"]).sum()
max_size = min(more_than_50k, less_than_50k)

# print(more_than_50k, less_than_50k)
# More elements with less than $50k income

data_1 = data[data["income"].isin(["<=50K"])].sample(n=max_size, random_state=seed1)
data_2 = data[data["income"].isin([">50K"])].sample(n=max_size, random_state=seed1)
temp = [data_1, data_2]
data = pd.concat(temp)

# print(data_filtered.shape)


######

# =================================== DATA STATISTICS =========================================== #

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method

######

# 2.5 YOUR CODE HERE

# verbose_print(data_filtered.describe())

######

# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs
categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                     'relationship', 'gender', 'native-country', 'income']

for feature in categorical_feats:
    pass
    ######

    # 2.5 YOUR CODE HERE

    # pie_chart(data_filtered, feature)

    ######

# visualize the first 3 features using pie and bar graphs

######

# 2.5 YOUR CODE HERE

    # binary_bar_chart(data_filtered, feature)

######

# =================================== DATA PREPROCESSING =========================================== #

# we need to represent our categorical features as 1-hot encodings
# we begin by converting the string values into integers using the LabelEncoder class
# next we convert the integer representations into 1-hot encodings using the OneHotEncoder class
# we don't want to convert 'income' into 1-hot so let's extract this field first
# we also need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# don't forget to stitch continuous and cat features together

# NORMALIZE CONTINUOUS FEATURES

a = []
for feature in col_names:
    if feature not in categorical_feats:
        cont_feat_data = data[feature].values
        cont_feat_data = (cont_feat_data - np.mean(cont_feat_data))/np.std(cont_feat_data)
        a.append(cont_feat_data)

age = a[0]
fnlwgt = a[1]
edu_num = a[2]
cap_gain = a[3]
cap_loss = a[4]
hours_week = a[5]

cont_data = np.concatenate((age.reshape(-1, 1), fnlwgt.reshape(-1, 1), edu_num.reshape(-1, 1), cap_gain.reshape(-1, 1),\
                            cap_loss.reshape(-1, 1), hours_week.reshape(-1, 1)), axis=1)
cont_data = pd.DataFrame(cont_data)


# ENCODE CATEGORICAL FEATURES

label_encoder = LabelEncoder()
oneh_encoder = OneHotEncoder()

######

# Label Encode Categorical Features
workclass = label_encoder.fit_transform(data["workclass"])
race = label_encoder.fit_transform(data["race"])
education = label_encoder.fit_transform(data["education"])
mar_status = label_encoder.fit_transform(data["marital-status"])
occupation = label_encoder.fit_transform(data["occupation"])
relationship = label_encoder.fit_transform(data["relationship"])
gender = label_encoder.fit_transform(data["gender"])
native_country = label_encoder.fit_transform(data["native-country"])
income = label_encoder.fit_transform(data["income"])

# Make a DataFrame of categorical features
d_categorical = {
    "workclass": workclass,
    "race": race,
    "education": education,
    "marital-status": mar_status,
    "occupation": occupation,
    "relationship": relationship,
    "gender": gender,
    "native-country": native_country
}

# One hot encode the categorical features
d_categorical = pd.DataFrame(d_categorical)
d_categorical = pd.DataFrame(oneh_encoder.fit_transform(d_categorical).toarray())

# Merge dataframes to create the final filtered dataframe
df_new = pd.concat([cont_data, d_categorical], axis=1)
# print(df_new.shape)


######
# Hint: .toarray() converts the DataFrame to a numpy array


# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter

######

# 2.7 YOUR CODE HERE

arr_new = np.array(df_new)
income = np.array(income)
training_features, validation_features, training_income, validation_income = \
    train_test_split(arr_new, income, test_size=0.20, random_state=seed2)


######

# =================================== LOAD DATA AND MODEL =========================================== #


def load_data(batchsize):
    ######

    # 3.2 YOUR CODE HERE

    training_data = AdultDataset(training_features, training_income)
    validation_data = AdultDataset(validation_features, validation_income)

    train_loader = DataLoader(training_data, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=batchsize)

    ######

    return train_loader, val_loader


def load_model(learn_rate, activation):

    ######

    # 3.4 YOUR CODE HERE

    model = MultiLayerPerceptron(103, activation)
    loss_fnc = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

    ######

    return model, loss_fnc, optimizer


def evaluate(model, val_loader):
    total_corr = 0

    ######

    # 3.6 YOUR CODE HERE

    for i, vbatch in enumerate(val_loader):
        feats, label = vbatch
        prediction = model(feats.float())
        corr = np.array((prediction > 0.5).squeeze().long() == label)
        total_corr += int(corr.sum())

    ######

    return float(total_corr)/len(val_loader.dataset)


def main(batch_size, learning_rate, num_epochs, eval_every, activation):

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', type=int)
    # parser.add_argument('--lr', type=float)
    # parser.add_argument('--epochs', type=int, default=20)
    # parser.add_argument('--eval_every', type=int, default=10)

    # args = parser.parse_args()

    ######

    # 3.5 YOUR CODE HERE

    t = 0
    i = 0
    model, loss_fnc, optimizer = load_model(learning_rate, activation)
    train_loader, val_loader = load_data(batch_size)

    batches = []
    # times = []
    valid_accuracy = []
    train_accuracy = []

    time0 = time()

    for epoch in range(num_epochs):
        tot_loss = 0
        tot_corr = 0

        for j, batch in enumerate(train_loader):
            feats, label = batch
            optimizer.zero_grad()
            predictions = model(feats.float())

            batch_loss = loss_fnc(input=predictions.squeeze(), target=label.float())
            tot_loss += batch_loss
            batch_loss.backward()
            optimizer.step()

            corr = ((predictions > 0.5).squeeze().long() == label)
            tot_corr += int(corr.sum())

            # evaluate model on the validation set every eval_every steps
            valid_acc = evaluate(model, val_loader)
            valid_accuracy.append(float(valid_acc))

            train_acc = evaluate(model, train_loader)
            train_accuracy.append(float(train_acc))

            # time1 = time() - time0
            # times.append(time1)

            batches.append(i)

            if (t + 1) % eval_every == 0:
                print("Epoch: {}, Step {} | Loss: {}| Test acc: {}".format(epoch + 1, t + 1, tot_loss / 100, valid_acc))
                tot_loss = 0

            t += 1
            i += 1
            # print("Train acc:{}".format(float(tot_corr) / len(train_loader.dataset)))
    '''
    plt.figure()
    plt.axes()
    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy")
    plt.plot(times, train_accuracy, label="Train")
    plt.plot(times, valid_accuracy, label="Validation")
    plt.savefig("batch_size"+str(batch_size)+"time.png")
    plt.show()
    '''

    print(activation + " time:", time()-time0)
    return batches, train_accuracy, valid_accuracy

    ######


if __name__ == "__main__":

    # batch_size = [64, 17932]
    # learning_rate = 0.1
    # num_epochs = [10, 1000]

    activation = ["tanh", "relu", "sigmoid"]

    batches = []
    train_results = []
    valid_results = []

    for i in range(len(activation)):
        batch, train_accuracy, valid_accuracy = main(batch_size, learning_rate, num_epochs, eval_every, activation[i])
        batches.append(batch)
        train_results.append(train_accuracy)
        valid_results.append(valid_accuracy)

    plt.figure()
    plt.axes()
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.plot(batches[0], train_results[0], label="Training Accuracy Tanh")
    plt.plot(batches[0], valid_results[0], label="Validation Accuracy Tanh")
    plt.plot(batches[1], train_results[1], label="Training Accuracy Relu")
    plt.plot(batches[1], valid_results[1], label="Validation Accuracy Relu")
    plt.plot(batches[2], train_results[2], label="Training Accuracy Sigmoid")
    plt.plot(batches[2], valid_results[2], label="Validation Accuracy Sigmoid")
    plt.savefig("activation_functions.png")
    plt.show()

    '''
    batch, train_accuracy, valid_accuracy = main(batch_size, learning_rate, num_epochs, eval_every, "tanh")
    plt.figure()
    plt.axes()
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.plot(batch, train_accuracy)
    plt.savefig("1.png")
    plt.show()

    plt.figure()
    plt.axes()
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.plot(batch, valid_accuracy)
    plt.savefig("2.png")
    plt.show()
    '''