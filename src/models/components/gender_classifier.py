import torch
from torch import nn


class GenderClassifier(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        output_size: int = 2,
        num_of_features: int = 2048,
    ) -> None:
        """Initialize a `ColorClassifier` module.

        :param input_size: The number of input features.
        :param output_size: The number of output features of the final linear layer.
        :num_of_features: The number of features from the backbone.
        """
        super().__init__()
        self.output_size = output_size
        self.classifier = nn.Linear(num_of_features, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        # use the pretrained model
        x = self.classifier(representation)
        x = self.sigmoid(x)
        return x.squeeze()


if __name__ == "__main__":
    _ = GenderClassifier()
