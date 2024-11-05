import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ResNetDepth(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.encoder_layers = list(resnet.children())[:-2]
        self.encoder = nn.Sequential(*self.encoder_layers)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    model = ResNetDepth(input_channels=3)
    input_tensor = torch.randn(1, 3, 224, 224)
    depth_output = model(input_tensor)
    print("Output shape:", depth_output.shape)