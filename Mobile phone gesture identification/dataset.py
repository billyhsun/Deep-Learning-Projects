'''
    Extend the torch.utils.data.Dataset class to build a GestureDataset class.
'''

import torch.utils.data as data


class DatasetClass(data.Dataset):

    def __init__(self, feats, labels):
        self.features = feats
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


