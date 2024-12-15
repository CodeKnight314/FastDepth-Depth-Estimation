import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from torchvision.models import ResNet18_Weights, ResNet34_Weights

class ResNetEncoder(nn.Module):
    """
    ResNet encoder that extracts features at multiple scales for the decoder.
    Configurable to ResNet-18 or ResNet-34.
    """
    def __init__(self, resnet_type='resnet18', input_channels=3):
        super(ResNetEncoder, self).__init__()
        assert resnet_type in ['resnet18', 'resnet34'], "resnet_type must be 'resnet18' or 'resnet34'."

        # Load pretrained model
        if resnet_type == 'resnet18':
            self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.resnet = models.resnet34(weight=ResNet34_Weights.DEFAULT)

        # Adjust first layer if input_channels != 3
        if input_channels != 3:
            old_weights = self.resnet.conv1.weight.data
            if input_channels == 1:
                # average across input channels
                new_weights = old_weights.mean(dim=1, keepdim=True)
            else:
                raise ValueError("Only 1 or 3 input channels are supported.")

            self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.resnet.conv1.weight.data = new_weights

        # Keep references to layers we need
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.conv1 = self.resnet.conv1
        self.maxpool = self.resnet.maxpool

    def forward(self, x):
        # Instead of including maxpool in x0, we only go up to relu:
        # Input: B,C,H,W

        # x0 after initial conv, bn, relu (no maxpool) -> Scale: H/2, W/2
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)  # (B,64,H/2,W/2)

        # Now apply maxpool and layer1 to get x1 -> Scale: H/4, W/4
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)   # (B,64,H/4,W/4)

        # Further layers
        x2 = self.layer2(x1)   # (B,128,H/8,W/8)
        x3 = self.layer3(x2)   # (B,256,H/16,W/16)
        x4 = self.layer4(x3)   # (B,512,H/32,W/32)

        return x0, x1, x2, x3, x4


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, upsample=True):
        super(UpConvBlock, self).__init__()
        self.upsample = upsample
        if self.upsample:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x


class Decoder(nn.Module):
    def __init__(self, enc_channels=[64, 64, 128, 256, 512], out_channels=1):
        super(Decoder, self).__init__()

        # x4:1/32 -> x3:1/16
        self.up4 = UpConvBlock(in_channels=enc_channels[4], out_channels=enc_channels[3])
        self.iconv4 = nn.Sequential(
            nn.Conv2d(enc_channels[3]*2, enc_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[3]),
            nn.ReLU(inplace=True)
        )

        # x3:1/16 -> x2:1/8
        self.up3 = UpConvBlock(in_channels=enc_channels[3], out_channels=enc_channels[2])
        self.iconv3 = nn.Sequential(
            nn.Conv2d(enc_channels[2]*2, enc_channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[2]),
            nn.ReLU(inplace=True)
        )

        # x2:1/8 -> x1:1/4
        self.up2 = UpConvBlock(in_channels=enc_channels[2], out_channels=enc_channels[1])
        self.iconv2 = nn.Sequential(
            nn.Conv2d(enc_channels[1]*2, enc_channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[1]),
            nn.ReLU(inplace=True)
        )

        # x1:1/4 -> x0:1/2
        self.up1 = UpConvBlock(in_channels=enc_channels[1], out_channels=enc_channels[0])
        self.iconv1 = nn.Sequential(
            nn.Conv2d(enc_channels[0]*2, enc_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[0]),
            nn.ReLU(inplace=True)
        )

        # x0:1/2 -> full resolution:1/1
        self.up0 = UpConvBlock(in_channels=enc_channels[0], out_channels=enc_channels[0]//2 if enc_channels[0]>1 else 1)
        self.iconv0 = nn.Sequential(
            nn.Conv2d((enc_channels[0]//2 if enc_channels[0]>1 else 1), out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x0, x1, x2, x3, x4):
        x = self.up4(x4)  # 1/32 -> 1/16
        x = torch.cat((x, x3), dim=1)
        x = self.iconv4(x)

        x = self.up3(x)   # 1/16 -> 1/8
        x = torch.cat((x, x2), dim=1)
        x = self.iconv3(x)

        x = self.up2(x)   # 1/8 -> 1/4
        x = torch.cat((x, x1), dim=1)
        x = self.iconv2(x)

        x = self.up1(x)   # 1/4 -> 1/2
        x = torch.cat((x, x0), dim=1)
        x = self.iconv1(x)

        x = self.up0(x)   # 1/2 -> 1/1
        x = self.iconv0(x)

        return x

class ResNetFastDepth(nn.Module):
    def __init__(self, resnet_type='resnet18', pretrained=True, input_channels=3):
        super(ResNetFastDepth, self).__init__()
        self.encoder = ResNetEncoder(resnet_type=resnet_type, pretrained=pretrained, input_channels=input_channels)
        enc_channels = [64, 64, 128, 256, 512]
        self.decoder = Decoder(enc_channels=enc_channels, out_channels=1)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)
        depth = self.decoder(x0, x1, x2, x3, x4)
        depth = torch.sigmoid(depth)
        return depth