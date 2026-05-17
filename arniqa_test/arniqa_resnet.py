import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import os
from typing import Tuple
from collections import OrderedDict

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ARNIQA_resnet():
    """
    ARNIQA model resnet implementation.

    This class implements the ARNIQA model for image quality assessment, which combines a ResNet50 encoder.
    """

    def __init__(self, device: torch.device = DEVICE):
        super().__init__()

        self.device = device

        self.encoder = torchvision.models.resnet50(weights=None).to(device)
        self.feat_dim = self.encoder.fc.in_features
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

        encoder_state_dict = torch.load(
            os.path.join(os.path.dirname(__file__), 'ARNIQA_resnet50.pth'),
            map_location=device
        )
        print("model loaded to device:", device)
        cleaned_encoder_state_dict = OrderedDict()
        for key, value in encoder_state_dict.items():
            # Remove the prefix
            if key.startswith('model.'):
                new_key = key[6:]
                cleaned_encoder_state_dict[new_key] = value

        self.encoder.load_state_dict(cleaned_encoder_state_dict)
        self.encoder.eval()

        self.default_mean = torch.Tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> float:
        """
        Forward pass of the ARNIQA model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            float: The predicted quality score.
        """
        x, x_ds = self._preprocess(x)

        f = F.normalize(self.encoder(x), dim=1)
        f_ds = F.normalize(self.encoder(x_ds), dim=1)
        f_combined = torch.hstack((f, f_ds)).view(-1, self.feat_dim * 2)
        return f, f_ds, f_combined

    def _preprocess(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Downsample the input image with a factor of 2 and normalize the original and downsampled images.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The normalized original and downsampled tensors.
        """
        x_ds = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        x_ds = (x_ds - self.default_mean.to(x_ds)) / self.default_std.to(x_ds)
        return x, x_ds
