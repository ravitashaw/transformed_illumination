import torch.nn as nn
import torch


class TransformerEmbed(nn.Module):
    """
    This class will take a set of input embeddings and use a transformer to produce a single output embedding
    """

    def __init__(self, num_heads, d_model, num_layers, feedforward_dim, reduction='mean'):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_layers = num_layers
        self.feedforward_dim = feedforward_dim
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,
                                                     nhead=num_heads,
                                                     dim_feedforward=feedforward_dim),
            num_layers=num_layers,
        )
        self.reduction = reduction

    def forward(self, x):
        # x dim should be: B x I x Z
        # where: B = batch size
        #        I = number of images
        #        Z = dimensionality of embedded vector
        # output should be: B x H
        # where: H = hidden feedforward dim
        output_embeddings = self.encoder(x)
        if self.reduction == 'mean':
            # avg across all images
            output = torch.mean(output_embeddings, dim=1)
        else:
            output = torch.sum(output_embeddings, dim=1)
        return output
