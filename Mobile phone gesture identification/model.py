'''
    Write a model for gesture classification.
'''

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    # input x is of shape (batch_size, 6, 100)

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(6, 12, 5)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(12, 24, 5)
        self.dropout = nn.Dropout(p=0.2)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(24, 48, 5)
        self.pool3 = nn.MaxPool1d(2)
        # self.conv4 = nn.Conv1d(25, 15, 5)
        # self.conv5 = nn.Conv1d(15, 10, 5)

        self.fc1 = nn.Linear(432, 256)
        self.fc1_bs = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.fc2_bs = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 64)
        self.fc3_bs = nn.BatchNorm1d(64)

        self.fc4 = nn.Linear(64, 26)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.pool3(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, x.shape[1] * x.shape[2])

        x = F.relu(self.fc1_bs(self.fc1(x)))
        x = F.relu(self.fc2_bs(self.fc2(x)))
        x = F.relu(self.fc3_bs(self.fc3(x)))
        x = self.fc4(x)
        # x = F.log_softmax(x)

        return x
