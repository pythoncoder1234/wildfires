import re

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.nn import LSTM, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

seq_length = 20
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super(SimpleLSTM, self).__init()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        lstm_out, hidden = self.lstm(input.view(1, 1, -1), hidden)
        output = self.fc(lstm_out.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train(model, train_loader, criterion, optimizer, num_epochs):
    train_loss = []

    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs, _ = model(inputs, None)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss.append(total_loss)

        # Print the average loss for this epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader)}')

    return train_loss

with open("alllines.txt") as f:
    lines = f.readlines()

lines = list(map(lambda line: line[1:-2], lines))
everything = " ".join(lines).replace("--", " ").replace("\t", "")
everything = re.sub('\S+', lambda m: re.sub('^\W+|\W+$', '', m.group()), everything).lower()
text = everything.split(" ")
words = np.unique(text)
print("Text parsed")

vector = OneHotEncoder().fit_transform(words.reshape(-1, 1)).toarray()
encoding_dict = {word: encoding for word, encoding in zip(words, vector)}
x = []
y = []

for i in range(len(text) - seq_length - 1):
    x.append(list(map(lambda word: encoding_dict[word], text[i:i + seq_length])))
    y.append(encoding_dict[text[i+seq_length]])

print("Data loader")
data_loader = DataLoader(CustomDataset(x, y), batch_size=10)
print("Preprocessing completed")
model = SimpleLSTM(seq_length, seq_length, 1)
train(model, data_loader, CrossEntropyLoss(), torch.Adam(), 5)
