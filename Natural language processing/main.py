import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy
import argparse
import os
from models import *


def load_model(lr, vector, embedding_dim, hidden_dim, model_type, num_filters, filter_sizes):
    loss_fnc = torch.nn.BCELoss()

    if model_type == "baseline":
        model = Baseline(embedding_dim, vector)
    elif model_type == "rnn":
        model = RNN(embedding_dim, vector, hidden_dim)
    elif model_type == "cnn":
        model = CNN(embedding_dim, vector, num_filters, filter_sizes)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, loss_fnc, optimizer


def evaluate(model, val_loader, loss_fnc):
    total_corr = 0
    num = 0
    batch_loss = 0
    for i, vbatch in enumerate(val_loader):
        feats = vbatch.text[0].transpose(0, 1)
        lengths = vbatch.text[1]
        label = vbatch.label
        prediction = model(feats, lengths)
        batch_loss = loss_fnc(input=prediction.squeeze().float(), target=label.float())
        corr = (prediction > 0.5).squeeze().float() == label.float()
        total_corr += int(corr.sum())
        num += label.shape[0]

    return float(total_corr)/num, batch_loss


def main(args):
    ######

    # 3.2 Processing of the data

    text = data.Field(sequential=True, include_lengths=True, tokenize='spacy')
    label = data.Field(sequential=False, use_vocab=False)

    train, val, test = data.TabularDataset.splits(path="./data/", train='train.tsv', validation='validation.tsv',
                                                  test='test.tsv', format='tsv', skip_header=True,
                                                  fields=[('', None), ('text', text), ('label', label)])
    train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(datasets=(train, val, test),
                                                                             sort_within_batch=True, repeat=False,
                                                                             batch_sizes=(args.batch_size,
                                                                                          args.batch_size, 1000),
                                                                             sort_key=lambda x: len(x.text))
    text.build_vocab(train)
    text.vocab.load_vectors(torchtext.vocab.GloVe(name="6B", dim="100"))
    filter_size = [2, 4]

    # 5 Training and Evaluation

    model, loss_fnc, optimizer = load_model(args.lr, text.vocab, args.emb_dim, args.rnn_hidden_dim, args.model,
                                            args.num_filt, filter_size)

    t = 0
    max_val = 0
    min_loss = 10
    for epoch in range(args.epochs):
        accum_loss = 0
        tot_corr = 0
        num_feats = 0

        for i, batch in enumerate(train_iterator):
            feats = batch.text[0].transpose(0, 1)
            lengths = batch.text[1]
            labels = batch.label

            optimizer.zero_grad()
            predictions = model(feats, lengths)

            batch_loss = loss_fnc(input=predictions.squeeze().float(), target=labels.float())
            accum_loss += batch_loss

            batch_loss.backward()
            optimizer.step()

            num_feats += labels.shape[0]
            corr = (predictions > 0.5).squeeze().float() == labels.float()
            tot_corr += int(corr.sum())

            if (t + 1) % 5 == 0:
                valid_acc, valid_loss = evaluate(model, val_iterator, loss_fnc)
                training_acc = float(tot_corr / num_feats)

                accum_loss = 0
                tot_corr = 0
                num_feats = 0

                if valid_acc > max_val:
                    max_val = valid_acc

                if valid_loss < min_loss:
                    min_loss = valid_loss

                print("Epoch: {}, Step: {}, Training Accuracy: {}, Training Loss: {}, Validation Accuracy: {}, "
                      "Validation Loss: {}".format(epoch + 1, t + 1, training_acc, batch_loss, valid_acc, valid_loss))

            t = t + 1

        test_acc, test_loss = evaluate(model, test_iterator, loss_fnc)
        print("Test Accuracy: {}, Test Loss: {}".format(test_acc, test_loss))
        print("Max Accuracy: {}, Min Loss: {}".format(max_val, min_loss))

    # torch.save(model, 'model_'+args.model+'.pt')

    ######


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='rnn',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    arguments = parser.parse_args()

    main(arguments)
