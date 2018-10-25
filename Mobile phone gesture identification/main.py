'''
    The entry into your code. This file should include a training function and an evaluation function.
'''

from csv2numpy import organize_data
from visualize_data import visualize_data
from bin_data import compute_stats
from normalize_data import normalize

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model import Net
from dataset import DatasetClass
from matplotlib import pyplot as plt


''' Hyper-parameters '''
seed = 42
learning_rate = 0.01     # Default = 0.01
batch_size = 32          # Default = 32
num_epochs = 200         # Default = 200
eval_every = 100         # Default = 100
test_split = 0.2         # Default = 0.2


''' Data Pre-processing '''
# 2.1 Parse data
data, labels = organize_data("unnamed_train_data")
print(data.shape)
print(labels.shape)

# 2.2 Understanding the data set
# visualize_data(data)

# 2.3 Basic statistics
# compute_stats(data)

# 2.4 Normalize data
normalized_arr = normalize(data)
print(normalized_arr.shape)

# 2.5 Train-validation split
# Use 50% training data, 50% validation data
training_features, validation_features, training_labels, validation_labels = \
    train_test_split(normalized_arr, labels, test_size=test_split, random_state=seed)

print(training_features.shape)
print(training_labels.shape)


# 3.1 PyTorch Dataset
def load_data(batchsize):
    training_data = DatasetClass(training_features, training_labels)
    validation_data = DatasetClass(validation_features, validation_labels)

    train_loader = DataLoader(training_data, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=batchsize, shuffle=True)

    return train_loader, val_loader


def load_model(learn_rate):
    model = Net()
    loss_fnc = torch.nn.CrossEntropyLoss()       # Subject to change
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)      # Subject to change

    return model, loss_fnc, optimizer


def evaluate(model, loader):
    total_corr = 0

    for j, batch in enumerate(loader):
        feats, label = batch
        prediction = model(feats.float())
        corr = (torch.argmax(prediction, dim=1) == torch.argmax(label, dim=1))
        total_corr += int(corr.sum())

    return float(total_corr)/len(loader.dataset)


def main(batch_size, learning_rate, num_epochs, eval_every):
    model, loss_fnc, optimizer = load_model(learning_rate)
    train_loader, val_loader = load_data(batch_size)

    batches = []
    valid_accuracy = []
    train_accuracy = []
    t = 0
    k = 0

    for epoch in range(num_epochs):
        tot_loss = 0
        tot_corr = 0

        for j, batch in enumerate(train_loader):
            feats, label = batch
            optimizer.zero_grad()
            predictions = model(feats.float())

            norm_labels = torch.argmax(label, dim=1)
            batch_loss = loss_fnc(input=predictions.squeeze(), target=norm_labels.long())
            tot_loss += batch_loss
            batch_loss.backward()
            optimizer.step()

            corr = (torch.argmax(predictions, dim=1) == torch.argmax(label, dim=1))
            tot_corr += int(corr.sum())

            if (t + 1) % eval_every == 0:
                valid_acc = evaluate(model, val_loader)
                valid_accuracy.append(float(valid_acc))

                train_acc = evaluate(model, train_loader)
                train_accuracy.append(float(train_acc))

                batches.append(k*eval_every)
                print("Epoch: {}, Step {} | Loss: {}| Valid acc: {}".format(epoch + 1, t + 1, tot_loss/100, valid_acc))
                tot_loss = 0

            t += 1
            k += 1

    plt.figure()
    plt.axes()
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.plot(batches, train_accuracy, label="Train")
    plt.plot(batches, valid_accuracy, label="Validation")
    plt.savefig("batch_size"+str(batch_size)+"time.png")
    plt.show()

    torch.save(model, 'model.pt')
    return model


def test(model):
    test_unnorm = np.load("test_data.npy")
    test_norm = normalize(test_unnorm)
    test_data = DatasetClass(test_norm, np.zeros([test_norm.shape[0], 1]))
    test_loader = DataLoader(test_data, batch_size=test_norm.shape[0], shuffle=False)
    pred = np.zeros([test_norm.shape[0]])
    for i, batch in enumerate(test_loader):
        feats, label = batch
        pred = model(feats.float())

    predictions = []
    for i in range(pred.shape[0]):
        predictions.append(torch.argmax(pred[i]))
    return np.array(predictions)


model = main(batch_size, learning_rate, num_epochs, eval_every)
prediction = test(model)
np.savetxt("predictions2.txt", prediction)
