import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

def parse_text():
    with open("alllines.txt") as f:
        lines = f.readlines()

    lines = list(map(lambda line: line.replace('"', ''), lines))
    everything = "".join(lines).lower()
    everything = "".join(filter(lambda x: x in characters, everything))
    print("Text parsed")
    return everything

def preprocess_text():
    with open("everything.txt") as f:
        everything = f.read()

    encoding_dict: dict[str, int] = {character: i for i, character in enumerate(characters)}
    x = np.zeros((len(everything), seq_length, len(characters)), dtype=bool)
    y = np.zeros((len(everything) - 1, len(characters)), dtype=bool)

    for i in range(len(everything) - seq_length - 1):
        for j in range(seq_length):
            character = everything[i + j]
            x[i, j, encoding_dict[character]] = 1
        y[i, encoding_dict[everything[i + seq_length + 1]]] = 1

    return x, y

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # input_size = hidden_size = seq_length = 40
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        global inputs1
        inputs1 = inputs
        lstm_out, hidden = self.lstm(inputs, hidden)
        output = self.fc(lstm_out.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx].astype(np.float32), self.labels[idx].astype(np.float32)


def train(model, train_loader, criterion, optimizer, num_epochs):
    global inputs
    train_loss = []

    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = torch.FloatTensor(inputs)
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

special_chars = list("!?.,'\n ")
characters = np.array([chr(i) for i in range(97, 97 + 26)] + special_chars)

# x, y = preprocess_text()
x, y = np.load("inputs.npy"), np.load("outputs.npy")

seq_length = 40

print("Preprocessing completed")
print("Data loader")
data_loader = DataLoader(CustomDataset(x, y), batch_size=10)
model = SimpleLSTM(seq_length, seq_length, 1)
train(model, data_loader, CrossEntropyLoss(), Adam(model.parameters(), lr=0.01), 5)
