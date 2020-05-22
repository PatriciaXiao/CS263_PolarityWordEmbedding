import torch.nn as nn
import torch.nn.functional as F


class NaiveClassify(nn.Module):
    def __init__(self, input_dim, hidden_size, n_class):
        super(NaiveClassify, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_class)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, outputs, labels):
        return self.criterion(outputs, labels)
