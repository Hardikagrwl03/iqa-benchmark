from json import encoder

from dotmap import DotMap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import os
from typing import Tuple
from collections import OrderedDict

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ARNIQA_mobilenet():
    """
    ARNIQA model mobilenet implementation.
    This class implements the ARNIQA model for image quality assessment, which combines a MobileNet encoder.
    """

    def __init__(self, device: torch.device = DEVICE):
        super().__init__()

        self.device = device
        self.model = Mobilenet(embedding_dim=128, pretrained=True, use_norm=True).to(device)
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'ARNIQA_mobilenet.pth')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        simclr = SimCLR({}, inference=True).to(device)
        simclr.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.mobilenet = simclr.encoder
        self.mobilenet.eval()
    
    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            f, f_norm = self.mobilenet(x)
        return f, f_norm
    
class SimCLR(nn.Module):
    def __init__(self, encoder_params: DotMap, temperature: float = 0.1, inference = False):
        super().__init__()
        if inference:
            self.encoder = Mobilenet(embedding_dim=128, pretrained=False)
        else:
            self.encoder = Mobilenet(embedding_dim=encoder_params.embedding_dim, pretrained=encoder_params.pretrained, use_norm=encoder_params.use_norm)
        self.temperature = temperature

    def forward(self, im_q, im_k=None):
        q, proj_q = self.encoder(im_q)
        return q, proj_q      

class Mobilenet(nn.Module):
    """
    ARNIQA model mobilenet implementation.

    This class implements the ARNIQA model for image quality assessment, which combines a MobileNet encoder.
    """

    def __init__(self, embedding_dim: int, pretrained: bool = True, use_norm: bool = True):
        super().__init__()

        self.use_norm = use_norm
        self.pretrained = pretrained
        self.embedding_dim = embedding_dim
        weights = None
        base = torchvision.models.mobilenet_v3_small(weights=weights)
        self.feat_dim = base.classifier[0].in_features
        self.model = nn.Sequential(*list(base.children())[:-1])

        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.embedding_dim)
        )

    def forward(self, x):
        f = self.model(x)
        f = f.view(-1, self.feat_dim)
        if self.use_norm:
            f = F.normalize(f, dim=1)
        g = self.projector(f)
        if self.use_norm:
            return f, F.normalize(g, dim=1)
        else:
            return f, g