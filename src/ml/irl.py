import torch
from torch.nn import functional as F
from functools import reduce
import operator


class FeatureExtractor(torch.nn.Module):
    # in_shape = 64x64x1
    def __init__(self, im_shape):
        super(FeatureExtractor, self).__init__()
        self.out_units = 128
        self.conv1 = torch.nn.Conv2d(1, 64, 5, 1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, 1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)
        self.conv4 = torch.nn.Conv2d(64, 64, 5, 1)
        # self.fc1 = torch.nn.Linear(7744, self.out_units)
        # self.fc2 = torch.nn.Linear(self.out_units, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.nn.MaxPool2d(2)(x)
        x = F.relu(self.conv3(x))
        x = torch.nn.MaxPool2d(2)(x)
        x = F.relu(self.conv4(x))
        x = torch.nn.MaxPool2d(2)(x)
        return x


t = FeatureExtractor((64, 64))
z = torch.rand((1, 1, 64, 64))
print(t(z).shape)
# print(t.parameter_count())


class PolicyHead(torch.nn.Module):
    """
    We will have one policy head per possible goal in the environment
    or should it be a value head?
    """
    def __init__(self, in_features, actions):
        super(PolicyHead, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.q = torch.nn.Linear(128, actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.q(x)
        return out


# will not be used for now
class Discriminator(torch.nn.Module):
    def __init__(self, units):
        self.fc1 = torch.nn.Linear(units, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.r = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.r(x)
        return out


class FullArchitecture(torch.nn.Module):
    def __init__(self, im_shape, goals):
        self.fe1 = FeatureExtractor(im_shape)
        self.heads = torch.nn.ModuleList([
            PolicyHead() for _ in range(goals)
        ])
        self.disc = Discriminator(self.f1.out_units)
        pass


def parameter_count(module: torch.nn.Module):
    return sum(map(lambda x: reduce(operator.mul, x.shape, 1), module.parameters()))
