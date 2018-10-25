import torch.nn as nn
import torch


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size, activation):

        super(MultiLayerPerceptron, self).__init__()

        ######

        # 3.3 YOUR CODE HERE

        self.fc1 = nn.Linear(input_size, 50)    # 50
        # self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(50, 1)         # 50

        self.activation = activation

        ######

    def forward(self, features):
        ######

        # 3.3 YOUR CODE HERE

        x = self.fc1(features)

        if self.activation == "relu":
            x = torch.relu(x)
        if self.activation == "tanh":
            x = torch.tanh(x)
        if self.activation == "sigmoid":
            x = torch.sigmoid(x)

        # x = torch.relu(x)     # tanh activation function
        x = self.fc2(x)
        x = torch.sigmoid(x)  # sigmoid activation function

        # x = self.fc3(x)
        # x = torch.tanh(x)
        # x = self.fc4(x)
        # x = torch.sigmoid(x)
        return x

        ######
