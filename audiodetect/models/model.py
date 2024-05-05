import torch
from torch import nn
from torchvision.models import efficientnet


class EffNet(nn.Module):
    def __init__(self, model_type, n_classes, pretrained=True):
        super().__init__()

        if model_type == "efficientnet_b0":
            if pretrained:
                weights = efficientnet.EfficientNet_B0_Weights.DEFAULT
            else:
                weights = None
            self.base_model = efficientnet.efficientnet_b0(weights=weights)
        elif model_type == "efficientnet_b1":
            if pretrained:
                weights = efficientnet.EfficientNet_B1_Weights.DEFAULT
            else:
                weights = None
            self.base_model = efficientnet.efficientnet_b1(weights=weights)
        elif model_type == "efficientnet_b2":
            if pretrained:
                weights = efficientnet.EfficientNet_B2_Weights.DEFAULT
            else:
                weights = None
            self.base_model = efficientnet.efficientnet_b2(weights=weights)
        elif model_type == "efficientnet_b3":
            if pretrained:
                weights = efficientnet.EfficientNet_B3_Weights.DEFAULT
            else:
                weights = None
            self.base_model = efficientnet.efficientnet_b3(weights=weights)
        else:
            raise ValueError("model type not supported")

        self.base_model.classifier[1] = nn.Linear(
            self.base_model.classifier[1].in_features, n_classes, dtype=torch.float32
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = torch.cat([x, x, x], dim=3).permute(0, 3, 1, 2)
        return self.base_model(x)
