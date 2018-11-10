"""
    6. Testing on Your Own Sentence.

    You will write a python script that prompts the user for a sentence input on the command line, and prints the
    classification from each of the three models, as well as the probability that this sentence is subjective.

    An example console output:

        Enter a sentence
        What once seemed creepy now just seems campy

        Model baseline: subjective (0.964)
        Model rnn: subjective (0.999)
        Model cnn: subjective (1.000)

        Enter a sentence
"""

from main import *
import torch


def tokenize(text):
    return [tok.text for tok in nlp.tokenizer(text)]


nlp = spacy.load('en')
text = data.Field(sequential=True, include_lengths=True, tokenize='spacy')
train, val, test = data.TabularDataset.splits(
    path='./data/', train='train.tsv',
    validation='validation.tsv', test='test.tsv', format='tsv',
    fields=[('', None), ('text', text)], skip_header=True)

baseline = torch.load('model_baseline.pt')
rnn = torch.load("model_rnn.pt")
cnn = torch.load('model_cnn.pt')

text.build_vocab(train, vectors="glove.6B.100d")
vocab = text.vocab

while True:
    sentence = input("Enter a sentence: ")
    tokens = tokenize(sentence)
    indices = torch.tensor([vocab.stoi[tok] for tok in tokens]).unsqueeze(dim=0)
    baseline_pred = baseline(indices, [len(indices)]).squeeze()
    rnn_pred = rnn(indices, [len(indices)]).squeeze()
    cnn_pred = cnn(indices, [len(indices)]).squeeze()

    baseline_pred = round(float(baseline_pred), 3)
    rnn_pred = round(float(rnn_pred), 3)
    cnn_pred = round(float(cnn_pred), 3)

    print("Model baseline: {} ({})".format("subjective" if baseline_pred > 0.5 else "objective", baseline_pred))
    print("Model rnn: {} ({})".format("subjective" if rnn_pred > 0.5 else "objective", rnn_pred))
    print("Model cnn: {} ({})".format("subjective" if cnn_pred > 0.5 else "objective", cnn_pred))

    response = input("Continue? Y/N: ")
    if response == 'Y' or response == 'y':
        continue
    else:
        break
