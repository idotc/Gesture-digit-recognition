import torch
import torch.nn as nn


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.hidden1(x)
        out = self.relu(out)
        out = self.hidden2(out)
        out = self.relu(out)
        out = self.predict(out)
        return out
