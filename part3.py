import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB
from collections import Counter


# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.LSTM = tnn.LSTM(50, hidden_size=100, batch_first=True, dropout=.4, num_layers=2, bidirectional=True)
        self.fc1 = tnn.Linear(100, 64)
        self.fc2 = tnn.Linear(64, 1)
        self.sigmoid = tnn.Sigmoid()
        self.drop_out = tnn.Dropout(0.4)

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        Create the forward pass through the network.
        """
        out, (ht, ct) = self.LSTM(input)
        out = self.drop_out(out)
        chk = self.fc1(ht[-1])
        out = self.drop_out(out)
        out = tnn.functional.relu(chk)
        out = self.drop_out(out)
        out = self.fc2(out)

        return torch.flatten(out)


class PreProcessing():
    def pre(x):
        """Called after tokenization"""
        # remove stop words and numbers
        i = 0
        length = len(x)
        contra = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                  "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                  "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                  "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you",
                  "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                  "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                  "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
                  "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                  "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                  "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
                  "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                  "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                  "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                  "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                  "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
                  "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
                  "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                  "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                  "here's": "here is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                  "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                  "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                  "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
                  "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is",
                  "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                  "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                  "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                  "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
                  "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                  "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                  "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                  "you'll've": "you will have", "you're": "you are", "you've": "you have"}
        stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                     "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                     'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                     'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                     'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
                     'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                     'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                     'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
                     "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                     "won't", 'wouldn', "wouldn't"]
        signs = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=',
                 '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€',
                 '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►',
                 '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒',
                 '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸',
                 '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣',
                 '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
        while (i < length):
            if x[i].isdigit() == True or x[i] in stopwords or x[i] in signs:
                del (x[i])
                length = length - 1
            elif x[i] in contra.keys():
                val = contra[x[i]]
                x[i] = val
            i = i + 1

        return x

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return tnn.BCEWithLogitsLoss()


def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion = lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.005)  # Minimise the loss using the Adam algorithm.

    for epoch in range(10):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        net.eval()
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")


if __name__ == '__main__':
    main()