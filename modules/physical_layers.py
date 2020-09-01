import torch.nn as nn
import torch


class Illumination(nn.Module):
    """
    Module to perform weighted illumination using an input image and led weights
    no learnable parameters
    """

    def __init__(self, num_leds = 25):
        super().__init__()
        self.image_shape = [1, 28, 28]
        self.default_illumination = nn.Sequential(nn.Conv2d(num_leds, 1, 1, bias=False))
        self.num_leds = num_leds

    def forward(self, data_cubes, channel_weights):
        # data cubes are of shape: B x C x (H x W)
        # led_weights are of shape: B x num_leds
        # channel weights are: B x C x 1
        if channel_weights is not None:
            batch_size = data_cubes.shape[0]
            led_weights = channel_weights.view(batch_size, -1, 1)
            # this is a broadcast multiplication on the H and W sides
            weighted_channels = data_cubes * led_weights
            # reducing the output dimensionality
            output = torch.sum(weighted_channels, dim=1, keepdim=True).view([batch_size] + self.image_shape)
            return output, led_weights.view(-1, 1, self.num_leds)
        else:
            led_weights = self.default_illumination[0].weight.view(1, 1, self.num_leds).expand(len(data_cubes), 1, self.num_leds)
            output = self.default_illumination(data_cubes.view(-1, self.num_leds, self.image_shape[-2], self.image_shape[-1]))
            return output, led_weights
