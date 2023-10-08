import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

def process_fire(row):
    return np.array(np.nan_to_num(row))

merge = pd.read_csv("merge.csv")
merge = merge.sort_values(['longitude', 'latitude'])
grouped = merge.groupby('datetime')

wind = grouped['speed'].apply(np.array).to_list()
fire = np.array(grouped['Power'].apply(process_fire).to_list())
fire = normalize(fire)

from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

outputs = labels = None
def train(model, train_loader, criterion, optimizer, num_epochs):
    global outputs, labels
    losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        losses.append(total_loss)

        # Print the average loss for this epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader)}')


from torch.utils.data import DataLoader

data = np.hstack((fire, wind))
print(data.shape)

ds = CustomDataset(data[:-1], fire[1:])
dl = DataLoader(ds, batch_size=1, shuffle=True)
loss_func = nn.MSELoss()
mlp = MLP(data.shape[1], 5000, data.shape[1] // 2)
train(mlp, dl, loss_func, optim.SGD(mlp.parameters(), lr=1E-4), 5)

