import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ResNetDepth(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder_layers = list(resnet.children())[:-2]
        self.encoder = nn.Sequential(*self.encoder_layers)

        self.upconv4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.upconv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upconv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.output_layer = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.upsample(F.relu(self.upconv4(x)))
        x = self.upsample(F.relu(self.upconv3(x)))
        x = self.upsample(F.relu(self.upconv2(x)))
        x = self.upsample(F.relu(self.upconv1(x)))

        x = self.upsample(F.relu(self.output_layer(x)))

        return torch.sigmoid(x)

if __name__ == "__main__":
    model = ResNetDepth(input_channels=3)
    input_tensor = torch.randn(1, 3, 224, 224)
    depth_output = model(input_tensor)
    print("Output shape:", depth_output.shape)