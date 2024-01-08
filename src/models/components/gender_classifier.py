import torch
from torch import nn

from . import backbones as B
from .utils import get_named_function
import torchvision.models as models


NAMED_MODEL = get_named_function(B)


class GenderClassifier(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        backbone: str,
        output_size: int = 10,
    ) -> None:
        """Initialize a `ColorClassifier` module.

        :param input_size: The number of input features.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()
        self.backbone = models.resnet50(weights="DEFAULT")
        num_filters = self.backbone.fc.in_features
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        layers = list(self.backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        
        # use the pretrained model
        self.classifier = nn.Linear(num_filters, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    _ = GenderClassifier()
