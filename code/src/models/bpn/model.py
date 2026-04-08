"""BPN network: ResNet18 backbone with FC head predicting blur parameters.

Architecture from STRIVE Section 3.2:
- Input: (N+1) concatenated ROI images (reference + N neighbors)
- Backbone: ResNet18 (modified first conv for multi-image input)
- Head: GAP -> FC -> ReLU -> FC -> 4N parameters
- Output: (sigma_x, sigma_y, rho, w) per neighbor frame
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class BPN(nn.Module):
    """Background Prediction Network."""

    def __init__(self, n_neighbors: int = 3, pretrained: bool = True):
        """
        Args:
            n_neighbors: Number of target neighbor frames (N).
            pretrained: Use ImageNet-pretrained ResNet18 weights.
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.n_params = 4 * n_neighbors  # (sigma_x, sigma_y, rho, w) per frame

        # Load ResNet18 backbone
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # Modify first conv layer: 3*(N+1) input channels instead of 3
        in_channels = 3 * (n_neighbors + 1)
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                             padding=3, bias=False)
        # Initialize new conv: replicate pretrained weights across input groups
        with torch.no_grad():
            # Average pretrained weights and tile across input channels
            avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1))

        # Build feature extractor (everything except final FC)
        self.features = nn.Sequential(
            new_conv,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        # FC head: 512 -> 256 -> 4N
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, self.n_params)
        self._init_head()

    def _init_head(self):
        """Initialize head so the network starts producing near-identity blur.

        - fc1: standard Kaiming init (ReLU follows)
        - fc2: very small weights + zero bias so raw outputs start ~0.
          This means tanh/softplus operate in their linear regime where
          gradients flow freely. Otherwise large initial outputs saturate
          tanh and the network gets stuck predicting constant values.
        """
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.001)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3*(N+1), H, W) concatenated reference + neighbor ROIs.
                    Channel order: [ref_rgb, neighbor1_rgb, ..., neighborN_rgb]

        Returns:
            dict with keys:
                sigma_x: (B, N) blur spread x
                sigma_y: (B, N) blur spread y
                rho: (B, N) rotation angle
                w: (B, N) blend weight
        """
        feat = self.features(images)
        feat = self.gap(feat).flatten(1)  # (B, 512)
        params = self.fc2(F.relu(self.fc1(feat)))  # (B, 4N)

        # Reshape to (B, N, 4) then split
        params = params.view(-1, self.n_neighbors, 4)

        # Apply appropriate activations
        sigma_x = F.softplus(params[:, :, 0])       # positive
        sigma_y = F.softplus(params[:, :, 1])       # positive
        rho = torch.tanh(params[:, :, 2]) * 3.14159  # [-pi, pi]
        w = torch.tanh(params[:, :, 3])              # [-1, 1]

        return {
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "rho": rho,
            "w": w,
        }

