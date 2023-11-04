import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np

def process(row):
    return np.pad(np.nan_to_num(row), (3080 - len(row), 0))

merge = pd.read_csv("merge2.csv")
merge = merge.sort_values(['longitude', 'latitude'])
merge['datetime'] = pd.to_datetime(merge['datetime'])
merge = merge.loc[merge['datetime'].apply(lambda value: value.minute % 10 == 0)]
grouped = merge.groupby('datetime')

wind = grouped['speed'].apply(process).to_list()
fire = grouped['Power'].apply(process).to_list()
wind = normalize(wind) * 10
fire = normalize(fire) * 10

from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_count):
        super(MLP, self).__init__()

        feature_cnt = np.linspace(input_size, output_size, hidden_count, dtype=int)
        self.layers = nn.ModuleList([nn.Linear(feature_cnt[i], feature_cnt[i + 1]) for i in range(len(feature_cnt) - 1)])
        print(self.layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.relu(x)

        return x


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    train_loss = []
    test_loss = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        loss, predicted = test(model, criterion, test_loader)
        test_loss.append(loss)

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss.append(total_loss / i)

        # Print the average loss for this epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training loss: {sum(train_loss) / len(train_loss):.5f}, Test loss: {sum(test_loss) / len(test_loss):.5f}')

    print("Test loss:", test(model, criterion, test_loader)[0])

    return train_loss, test_loss


inp = None
predicted = []
actual = []
def test(model, criterion, dl):
    global inp
    model.eval()
    total_loss = 0
    for i, (inputs, labels) in enumerate(dl):
        predicted.append(pr := model(inputs))
        actual.append(labels)
        loss = criterion(pr, labels)
        total_loss += loss.item()

    return total_loss / i, predicted


from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

data = np.hstack((fire, wind))
print("Data shape:", data.shape)

train_x, test_x, train_y, test_y = train_test_split(data[:-1], fire[1:], shuffle=False)
train_dl = DataLoader(CustomDataset(train_x, train_y), batch_size=10)
test_dl = DataLoader(CustomDataset(test_x, test_y), batch_size=10)

epochs = 10
loss_func = nn.MSELoss()
mlp = MLP(data.shape[1], fire.shape[1], 5)
train_loss, test_loss = train(mlp, train_dl, test_dl, loss_func, optim.Adam(mlp.parameters(), lr=0.001), epochs)


def plot_loss(train_loss, test_loss):
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training loss")
    plt.plot(range(1, len(test_loss) + 1), test_loss, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss ($L2$)")
    plt.legend()
    plt.show()


# pred = predicted[-1][5].detach().numpy()
# act = actual[-1][5].detach().numpy()
# np.savetxt("predicted.txt", pred)
# np.savetxt("actual.txt", act)
plot_loss(train_loss, test_loss)
