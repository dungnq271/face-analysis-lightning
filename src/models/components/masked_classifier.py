import torch
from torch import nn


class MaskedClassifier(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        output_size: int = 2,
    ) -> None:
        """Initialize a `ColorClassifier` module.

        :param input_size: The number of input features.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()
        self.output_size = output_size
        self.classifier = nn.Linear(2048, self.output_size)
        self.softmax = nn.Softmax()

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        # use the pretrained model
        x = self.classifier(representation)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    _ = MaskedClassifier()
