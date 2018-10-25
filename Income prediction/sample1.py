import pandas as pd
from util import *

# 2.1 bring data in DataFrame object
data = pd.read_csv("./data/adult.csv")

# 2.2 Look at data
# The 'shape' is the dimensions of the table
print("Shape: ", data.shape)
# Print the names of the columns - comes from first line of spreadsheet
print(data.columns)

# Print the first 5 columns
verbose_print(data.head())
# print out the entire column - see can reference by the column head name
print(data["education"])

# 2.3 missing data
# determine if any given entry in a field is a "?" - i.e. missing
print("is workclass field missing?", data["workclass"].isin(["?"]))

# count the number that is missing
print("number of missing workclass is ", data["workclass"].isin(["?"]).sum())

# 2.5 Visualization
# Let's look at the categorical features
categorical_feats = ['workclass', 'race', 'education', 'maritalstatus', 'occupation', 'income']

# Print out each feature's number of values
for feature in categorical_feats:
    print("feature: ", feature)
    print(data[feature].value_counts())
verbose_print(data.describe())

for i in range(3):
    binary_bar_chart(data, categorical_feats[i])
    pie_chart(data, categorical_feats[i])
"""
"""

