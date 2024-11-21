import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from torchvision.models import ResNet18_Weights, ResNet34_Weights

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = (kernel_size-1)//2

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)

class MConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1):
        super(MConvBlock, self).__init__()

        padding = (kernel_size-1)//2
        self.depth_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=input_channels),
            nn.BatchNorm2d(input_channels),
            nn.ReLU()
        )

        self.point_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class ResNetFastDepth(nn.Module):
    def __init__(self, layers: int, output_size: int, in_channels: int = 3, pretrained: bool = True, mode: str = "skip"):
        super().__init__()

        if layers not in [18, 34]:
            raise ValueError(f"Only 18 and 34 layers model are defined for ResNet. Got {layers}")

        self.output_size = output_size
        if pretrained:
            if layers == 18:
                weights = ResNet18_Weights.IMAGENET1K_V1
            elif layers == 34:
                weights = ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None

        model = models.__dict__[f'resnet{layers}'](weights=weights)
        if not pretrained:
            model.apply(weights_init)

        if in_channels == 3:
            self.conv1 = model._modules['conv1']
            self.bn1 = model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.relu = model._modules['relu']
        self.maxpool = model._modules['maxpool']
        self.layer1 = model._modules['layer1']
        self.layer2 = model._modules['layer2']
        self.layer3 = model._modules['layer3']
        self.layer4 = model._modules['layer4']

        if layers <= 34:
            num_channels = 512
        elif layers == 50:
            num_channels = 2048
            mode = None
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=1024, kernel_size=1, stride=1, padding=0)
        weights_init(self.conv2)

        kernel_size = 5
        if mode == "skip":
            self.decode_conv1 = ConvBlock(1024, 512, kernel_size)
            self.decode_conv2 = ConvBlock(512, 256, kernel_size)
            self.decode_conv3 = ConvBlock(256, 128, kernel_size)
            self.decode_conv4 = ConvBlock(128, 64, kernel_size)
            self.decode_conv5 = ConvBlock(64, 32, kernel_size)
            self.decode_conv6 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        elif mode == "concat":
            self.decode_conv1 = ConvBlock(1024, 512, kernel_size)
            self.decode_conv2 = ConvBlock(768, 256, kernel_size)
            self.decode_conv3 = ConvBlock(384, 128, kernel_size)
            self.decode_conv4 = ConvBlock(192, 64, kernel_size)
            self.decode_conv5 = ConvBlock(128, 32, kernel_size)
            self.decode_conv6 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        else:
            self.decode_conv1 = ConvBlock(1024, 512, kernel_size)
            self.decode_conv2 = ConvBlock(512, 256, kernel_size)
            self.decode_conv3 = ConvBlock(256, 128, kernel_size)
            self.decode_conv4 = ConvBlock(128, 64, kernel_size)
            self.decode_conv5 = ConvBlock(64, 32, kernel_size)
            self.decode_conv6 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)

        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

        self.mode = mode

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x2 = self.maxpool(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)

        x7 = self.conv2(x6)

        if self.mode == "skip":
            y10 = self.decode_conv1(x7)
            y9 = F.interpolate(y10 + x6, scale_factor=2, mode='nearest')
            y8 = self.decode_conv2(y9)
            y7 = F.interpolate(y8 + x5, scale_factor=2, mode='nearest')
            y6 = self.decode_conv3(y7)
            y5 = F.interpolate(y6 + x4, scale_factor=2, mode='nearest')
            y4 = self.decode_conv4(y5)
            y3 = F.interpolate(y4 + x3, scale_factor=2, mode='nearest')
            y2 = self.decode_conv5(y3 + x1)
            y1 = F.interpolate(y2, scale_factor=2, mode='nearest')
            y = self.decode_conv6(y1)
        elif self.mode == "concat":
            y10 = self.decode_conv1(x7)
            y9 = F.interpolate(y10, scale_factor=2, mode='nearest')
            y8 = self.decode_conv2(torch.cat((y9, x5), 1))
            y7 = F.interpolate(y8, scale_factor=2, mode='nearest')
            y6 = self.decode_conv3(torch.cat((y7, x4), 1))
            y5 = F.interpolate(y6, scale_factor=2, mode='nearest')
            y4 = self.decode_conv4(torch.cat((y5, x3), 1))
            y3 = F.interpolate(y4, scale_factor=2, mode='nearest')
            y2 = self.decode_conv5(torch.cat((y3, x1), 1))
            y1 = F.interpolate(y2, scale_factor=2, mode='nearest')
            y = self.decode_conv6(y1)
        else:
            y10 = self.decode_conv1(x7)
            y9 = F.interpolate(y10, scale_factor=2, mode='nearest')
            y8 = self.decode_conv2(y9)
            y7 = F.interpolate(y8, scale_factor=2, mode='nearest')
            y6 = self.decode_conv3(y7)
            y5 = F.interpolate(y6, scale_factor=2, mode='nearest')
            y4 = self.decode_conv4(y5)
            y3 = F.interpolate(y4, scale_factor=2, mode='nearest')
            y2 = self.decode_conv5(y3)
            y1 = F.interpolate(y2, scale_factor=2, mode='nearest')
            y = self.decode_conv6(y1)

        return y

class MobileNetFastDepth(nn.Module): 
    def __init__(self, input_channels: int = 3): 
        super().__init__() 
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU()
        )
        
        self.conv_stack = nn.ModuleList()
        self.conv_channels = [32, 64, 128, 128, 256, 256, 512]
        for i in range(6): 
            stride = 1 if i % 2 else 2
            self.conv_stack.append(MConvBlock(input_channels=self.conv_channels[i], output_channels=self.conv_channels[i+1], stride=stride))
        
        for i in range(5): 
            self.conv_stack.append(MConvBlock(512, 512, stride=1))
        
        self.conv_stack.append(MConvBlock(512, 1024, stride=2))
        self.conv_stack.append(MConvBlock(1024, 1024, 1))

        kernel_size = 5

        self.decode_conv1 = MConvBlock(input_channels=1024, output_channels=512, kernel_size=kernel_size)
        self.decode_conv2 = MConvBlock(input_channels=1024, output_channels=256, kernel_size=kernel_size)
        self.decode_conv3 = MConvBlock(input_channels=512, output_channels=128, kernel_size=kernel_size)  # Concatenate with x3
        self.decode_conv4 = MConvBlock(input_channels=256, output_channels=64, kernel_size=kernel_size)   # Concatenate with x2
        self.decode_conv5 = MConvBlock(input_channels=64, output_channels=32, kernel_size=kernel_size)    # Concatenate with x1
        self.decode_conv6 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        
        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def forward(self, x):
        x = self.initial_conv(x)
        for i, layer in enumerate(self.conv_stack):
            x = layer(x)
            print(i, x.shape)
            if i == 5: 
                x1 = x  # Save intermediate feature map
            elif i == 3: 
                x2 = x  # Save intermediate feature map
            elif i == 1: 
                x3 = x  # Save intermediate feature map

        x = self.decode_conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.decode_conv2(torch.cat([x1, x], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.decode_conv3(torch.cat([x2, x], dim=1))  # Concatenate with x3
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.decode_conv4(torch.cat([x3, x], dim=1))  # Concatenate with x2
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.decode_conv5(x)  # Concatenate with x1
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.decode_conv6(x)
        return x