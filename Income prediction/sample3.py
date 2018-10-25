import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn

from model import MultiLayerPerceptron
from dataset import AdultDataset
from util import *
#from plot import plot_graph
from matplotlib import pyplot as plt


""" Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""
seed = 0

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
#The 'shape' is the dimensions of the table
#print ("Shape: ", data.shape)

# Print the names of the columns - comes from first line of spreadsheet
#print (data.columns)

# Print the first 5 columns
#verbose_print (data.head())

# print out the entire column - see can reference by the column head name
#print (data["education"])

#x = (data["income"].value_counts())
#print("Low Income:", x["<=50K"])
#print("High Income:", x[">50K"])
#print(x)

######


# =================================== DATA CLEANING =========================================== #

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

# let's first count how many missing entries there are for each feature
col_names = data.columns
num_rows = data.shape[0]
oldshape = data.shape
for feature in col_names:

    ######

    total_missing = data[feature].isin(["?"]).sum()

    ######

# next let's throw out all rows (samples) with 1 or more "?"
# Hint: take a look at what data[data["income"] != ">50K"] returns
# Hint: if data[field] do not contain strings then data["income"] != ">50K" will return an error

    ######

    if total_missing > 0:
        data = data[data[feature] != "?"]

#print(data.shape)
#print('Columns left: ', data.shape[1] )
#print('Columns removed: ', oldshape[1] - data.shape[1] )
#print('Rows left: ', data.shape[0])
#print('Rows removed: ', oldshape[0] - data.shape[0] )


    ######

# =================================== BALANCE DATASET =========================================== #

    ######

data_lowincome = data[data["income"] == "<=50K"]
data_highincome = data[data["income"] == ">50K"]

new_size = min(data_lowincome.shape[0], data_highincome.shape[0])
#print(new_size)

data_lowincome = data_lowincome.sample(n = new_size, random_state = seed)
data_highincome = data_highincome.sample(n = new_size, random_state = seed)

data = pd.concat([data_lowincome, data_highincome])
#print(data.shape)

#print((data["income"].value_counts()))
#print(data)


    ######

# =================================== DATA STATISTICS =========================================== #

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method

######

######

# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs4

categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']

#for feature in categorical_feats:
    #print("feature: ",feature)
    #print(data[feature].value_counts())

#verbose_print(data.describe())
#for i in range(3):
    #binary_bar_chart(data, categorical_feats[i])
    #pie_chart(data, categorical_feats[i])

# visualize the first 3 features using pie and bar graphs

######

# 2.5 YOUR CODE HERE

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

cont_data = np.concatenate((age.reshape(-1,1), fnlwgt.reshape(-1,1), edu_num.reshape(-1,1), cap_gain.reshape(-1,1), cap_loss.reshape(-1,1), hours_week.reshape(-1,1)), axis = 1)
cont_data = pd.DataFrame(cont_data)


# ENCODE CATEGORICAL FEATURES
label_encoder = LabelEncoder()
oneh_encoder = OneHotEncoder()
######

#First Label Encode
workclass = label_encoder.fit_transform(data["workclass"])
race = label_encoder.fit_transform(data["race"])
education = label_encoder.fit_transform(data["education"])
mar_status = label_encoder.fit_transform(data["marital-status"])
occupation = label_encoder.fit_transform(data["occupation"])
relationship = label_encoder.fit_transform(data["relationship"])
gender = label_encoder.fit_transform(data["gender"])
native_country = label_encoder.fit_transform(data["native-country"])
income = label_encoder.fit_transform(data["income"])

#Then make a DataFrame of categorical features
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

#One hot encode the categorical features
d_categorical = pd.DataFrame(d_categorical)
d_categorical = pd.DataFrame(oneh_encoder.fit_transform((d_categorical)).toarray())

#Stitch everything back together
data_new = pd.concat([cont_data, d_categorical], axis = 1)
######
# Hint: .toarray() converts the DataFrame to a numpy array


# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter

######
features = np.array(data_new)
labels = np.array((income))

training_set_feat, validation_set_feat, training_set_labels, validation_set_labels\
    = train_test_split(features, labels, test_size = 0.2, random_state = seed)


"""
print(training_set_feat.shape)
print(validation_set_feat.shape)
print('\n')
print(training_set_labels.shape)
print(validation_set_labels.shape)
"""



######

# =================================== LOAD DATA AND MODEL =========================================== #

def load_data(batch_size):

    train_dataset = AdultDataset(training_set_feat, training_set_labels)
    validation_dataset = AdultDataset(validation_set_feat, validation_set_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle = True)

    return train_loader, val_loader


def load_model(lr):

    ######

    model = MultiLayerPerceptron(103)
    loss_fnc = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    ######

    return model, loss_fnc, optimizer


def evaluate(model, val_loader):
    total_corr = 0

    ######

    for i, vbatch in enumerate(val_loader):
        feats, label = vbatch
        feats = feats.float()
        outputs = model(feats)
        #print(label.shape == np.array((outputs > 0.5).squeeze().shape))
        corr = (np.array((outputs > 0.5).squeeze()) == (label > 0.5))
        total_corr += int(corr.sum())

    ######

    return float(total_corr)/len(val_loader.dataset)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=10)

    args = parser.parse_args()

    ######

    train_loader, val_loader = load_data(args.batch_size)
    model, loss_fnc, optimizer = load_model(args.lr)

    t = 0
    N = args.eval_every
    total_cor_plotting=0
    valid_acc_array = []
    train_acc_array = []

    for epoch in range(args.epochs):
        accum_loss = 0
        tot_corr = 0

        for i, batch in enumerate(train_loader):
            inputs, labels = batch

            optimizer.zero_grad()
            inputs = inputs.float()
            outputs = model(inputs)


            loss = loss_fnc(outputs.squeeze().float(), labels.float())
            loss.backward()
            optimizer.step()
            accum_loss += loss

            corr = ((outputs > 0.5).squeeze() == (labels > 0.5))
            #print(int(corr.sum()))
            tot_corr += int(corr.sum())


            #Evaluate
            valid_acc = evaluate(model, val_loader)
            valid_acc_array.append(valid_acc)

            train_acc = evaluate(model, train_loader)
            train_acc_array.append(train_acc)

            if (t + 1) % N == 0:
                valid_acc = evaluate(model, val_loader)
                print(
                    "Epoch: {}, Step {} | Loss: {}| Test acc:{}".format(epoch + 1, t + 1, accum_loss / N, valid_acc))

                accum_loss = 0

            t = t + 1

    # print("Train acc:{}".format(float(tot_corr) / len(train_loader.dataset)))










if __name__ == "__main__":
    main()