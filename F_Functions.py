import gpytorch
import torch.nn as nn


class Net1(nn.Module):  # Net1 has one hidden layer with Linear activation function

    def __init__(self, input_size, hidden_size, num_classes):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


class Net2(nn.Module):  # Net2 has two hidden layers with Linear activation function

    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class Net3(nn.Module):  # Net3 has three hidden layers with Linear activation function

    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(Net3, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=1)
        self.cov_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(),
                                                           num_tasks=1, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.cov_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, cov_x)
