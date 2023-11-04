import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import *
from sklearn.preprocessing import OneHotEncoder, normalize

from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_count):
        super(MLP, self).__init__()

        # feature_cnt = np.linspace(input_size, output_size, hidden_count, dtype=int)
        # self.layers = nn.ModuleList([nn.Linear(feature_cnt[i], feature_cnt[i + 1]) for i in range(len(feature_cnt) - 1)])
        self.layers = nn.ModuleList([nn.Linear(input_size if i == 0 else 512, 512 if i < hidden_count - 1 else output_size) for i in range(hidden_count)])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.relu(x)

        sm = self.softmax(x)
        return sm


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train(model, train_loader, test_loader, loss_func, optimizer, num_epochs):
    train_loss = []
    test_loss = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        loss, pr = test(model, loss_func, test_loader)
        test_loss.append(loss)

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss.append(total_loss / i)

        # Print the average loss for this epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training loss: {sum(train_loss) / len(train_loss):.5f}, Test loss: {sum(test_loss) / len(test_loss):.5f}')

    print("Test loss:", test(model, loss_func, test_loader)[0])

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

train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

train_x = train_df.iloc[:, 1:]
train_y = train_df.iloc[:, :1]
test_x = test_df.iloc[:, 1:]
test_y = test_df.iloc[:, :1]

oh = OneHotEncoder()
train_x, train_y, test_x, test_y = map(lambda df: df.values, [train_x, train_y, test_x, test_y])
train_x, test_x = map(lambda arr: normalize(arr), [train_x, test_x])
train_y, test_y = map(lambda arr: oh.fit_transform(arr).toarray(), [train_y, test_y])

train_dl = DataLoader(CustomDataset(train_x, train_y), batch_size=1000)
test_dl = DataLoader(CustomDataset(test_x, test_y), batch_size=1000)

epochs = 10
loss_func = nn.CrossEntropyLoss()
mlp = MLP(784, 10, 3)
train_loss, test_loss = train(mlp, train_dl, test_dl, loss_func, optim.Adam(mlp.parameters(), lr=0.001), epochs)

def plot_loss(train_loss, test_loss):
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training loss")
    plt.plot(range(1, len(test_loss) + 1), test_loss, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss ($L2$)")
    plt.legend()
    plt.show()


def metrics():
    pr, act = map(lambda tensor: np.array(list(map(lambda arr: np.argmax(arr)+1, tensor.detach().numpy()))), [predicted[-1], actual[-1]])
    accuracy = accuracy_score(act, pr) * 100
    acc_str = f"Accuracy: {accuracy:.1f}%"
    sns.set_theme()
    ax = sns.heatmap(confusion_matrix(act, pr), annot=True, fmt="d")
    ax.set_title(acc_str)
    plt.show()
    print(acc_str)


# pred = predicted[-1][5].detach().numpy()
# act = actual[-1][5].detach().numpy()
# np.savetxt("predicted.txt", pred)
# np.savetxt("actual.txt", act)
plot_loss(train_loss, test_loss)
metrics()
