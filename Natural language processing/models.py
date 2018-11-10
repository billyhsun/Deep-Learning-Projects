import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()
        
        ######
        
        # 4.1 YOUR CODE HERE
        
        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        self.embed_dim = embedding_dim
        self.fc1 = nn.Linear(embedding_dim, 1)
        
        ######

    def forward(self, x, lengths=None):
        ######

        # 4.1 YOUR CODE HERE

        x = self.embed(x)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = F.sigmoid(x)
        return x

        ######


class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()

        ######
        
        # 4.2 YOUR CODE HERE
        
        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        self.embed_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gru1 = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=self.num_layers)
        self.linear = nn.Linear(embedding_dim, 1)

        ######

    def forward(self, x, lengths):
        ######

        # 4.2 YOUR CODE HERE
        
        x = self.embed(x)
        # x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        output, h_n = self.gru1(x.transpose(0, 1))  # x or x.transpose(0, 1)
        x = h_n[0]
        x = self.linear(x)
        x = F.sigmoid(x)
        return x

        ######


class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()

        ######
        
        # 4.3 YOUR CODE HERE

        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        self.embed_dim = embedding_dim
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[0], embedding_dim))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[1], embedding_dim))
        self.linear = nn.Linear(embedding_dim, 1)
        
        ######

    def forward(self, x, lengths=None):
        ######

        x = self.embed(x)
        # print(x.shape)
        x1 = F.relu(self.conv1(x.reshape(-1, 1, x.shape[1], x.shape[2]))).squeeze(3)  # batch, channel, len of text - 1
        # print(x1.shape)
        x2 = F.relu(self.conv2(x.reshape(-1, 1, x.shape[1], x.shape[2]))).squeeze(3)
        # print(x2.shape)
        x1, pos = torch.max(x1, 2)
        x2, pos = torch.max(x2, 2)
        x = torch.cat((x1, x2), 1)
        
        return F.sigmoid(self.linear(x))

        ######
