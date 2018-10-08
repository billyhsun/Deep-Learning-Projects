import torch.utils.data as data


class AdultDataset(data.Dataset):

    def __init__(self, dataset, income):
        ######

        # 3.1 YOUR CODE HERE

        self.features = dataset  # dataset and income must be Pandas dataframe objects
        self.labels = income

        ######

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        ######

        # 3.1 YOUR CODE HERE

        return self.features[index], self.labels[index]

        ######
