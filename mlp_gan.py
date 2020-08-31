import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, in_features, out_features, n_hidden, hidden_size):
        """ MLP layers with no output excitation """
        super().__init__()
        layers = []
        x_in, x_out = in_features, hidden_size
        for _ in range(n_hidden):
            layers.append(nn.Linear(x_in, x_out))
            layers.append(nn.LeakyReLU(negative_slope=0.02, inplace=True))
            x_in = x_out
        x_out = out_features
        layers.append(nn.Linear(x_in, x_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = MLP(in_features=1, out_features=1, n_hidden=3, hidden_size=128)
        self.nonlin = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        out = self.nonlin(x)
        return out

    def sample_noise(self, batch_size):
        return torch.randn(batch_size, 1, dtype=torch.float32)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = MLP(in_features=1, out_features=1, n_hidden=3, hidden_size=128)
        self.nonlin = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        out = self.nonlin(x)
        return out
