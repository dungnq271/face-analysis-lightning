from typing import Tuple

import torch
from torch import nn

from . import backbones as B
from .utils import get_named_function
import torchvision.models as models


class FaceAttrsClassifier(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        backbone: str,
        race_output_size: int = 4,
        gender_output_size: int = 2,
        age_output_size: int = 6,
        skintone_output_size: int = 4,
        emotion_output_size: int = 7,
        masked_output_size: int = 2,
        dynamic_weights_loss: bool = False
    ) -> None:
        """Initialize a `ColorClassifier` module.

        :param input_size: The number of input features.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()
        if backbone == "resnet50":
            self.backbone = models.resnet50(weights="DEFAULT")
        elif backbone == "resnet101":
            self.backbone = models.resnet101(weights="DEFAULT")
        num_filters = self.backbone.fc.in_features

        # Freeze the backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        # for param in self.backbone.fc.parameters():
        #     param.requires_grad = True
        layers = list(self.backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model
        self.race_classifier = nn.Linear(num_filters, race_output_size)
        self.gender_classifier = nn.Linear(num_filters, gender_output_size)
        self.age_classifier = nn.Linear(num_filters, age_output_size)
        self.skintone_classifier = nn.Linear(num_filters, skintone_output_size)
        self.emotion_classifier = nn.Linear(num_filters, emotion_output_size)
        self.masked_classifier = nn.Linear(num_filters, masked_output_size)
        self.softmax = nn.Softmax(dim=1)

        self.dynamic_weights_loss = dynamic_weights_loss
        if dynamic_weights_loss:
            self.loss_weight_fc = nn.Linear(num_filters, 6)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        representations = self.feature_extractor(x).flatten(1)

        race_pred = self.race_classifier(representations)
        gender_pred = self.gender_classifier(representations)
        age_pred = self.age_classifier(representations)
        skintone_pred = self.skintone_classifier(representations)
        emotion_pred = self.emotion_classifier(representations)
        masked_pred = self.masked_classifier(representations)

        race_pred = self.softmax(race_pred)
        gender_pred = self.softmax(gender_pred)
        age_pred = torch.sigmoid(age_pred)
        skintone_pred = self.softmax(skintone_pred)
        emotion_pred = self.softmax(emotion_pred)
        masked_pred = self.softmax(masked_pred)

        loss_weights_pred = 0.0
        if self.dynamic_weights_loss:
            loss_weights_pred = self.softmax(self.loss_weight_fc(representations)).mean(dim=0)

        return race_pred, gender_pred, age_pred, skintone_pred, emotion_pred, masked_pred, loss_weights_pred


if __name__ == "__main__":
    _ = FaceAttrsClassifier()
