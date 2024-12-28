import torch.nn as nn
from torch.utils.data import Dataset

class TorchDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self): return self.y.shape[0]

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]


class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, dropout_flag: bool = False):
        """
        n_inputs (int): number of input features for the model.
        n_outputs (int): number of output classes for the model.
        dropout (bool): Whether to include dropout, default false.
        """
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 5)
        self.fc2 = nn.Linear(5, n_outputs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.dropout_flag = dropout_flag
    def forward(self, x):
        # Perform first fully connected layer operation followed by ReLU activation function
        x=self.fc1(x)
        x=self.relu(x)
        if self.dropout_flag:
            x=self.dropout(x)
        # Perform second fully connected layer
        x=self.fc2(x)
        return x

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
