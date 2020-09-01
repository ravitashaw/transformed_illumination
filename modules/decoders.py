import torch.nn as nn
import torch


class PhysicalDecoder(nn.Module):

    def __init__(self, num_leds, d_model):
        """
        Takes in an encoded state from transformer and outputs a physical parameterization for the physical layer
        num_l
        """
        super().__init__()
        self.num_leds = num_leds
        self.d_model = d_model
        self.std = 0.05

        self.model = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=num_leds)
            # nn.ReLU()
        )

    def forward(self, x):
        mu = self.model(x)
        return mu
        # noise = torch.randn_like(mu) * self.std
        # new_phi = torch.sigmoid(mu + noise)
        # return new_phi


class DecisionDecoder(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.model = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2)
        )

    def forward(self, x):
        return self.model(x)


class ClassificationDecoder(nn.Module):

    def __init__(self, d_model, num_classes):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=num_classes)
        )

    def forward(self, x):
        return self.model(x)