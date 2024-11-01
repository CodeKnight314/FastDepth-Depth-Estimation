import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        padding = (kernel_size)//2
        self.depth_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU()
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class MobileNetEncoder(nn.Module):
    def __init__(self, input_channels: int):
        super(MobileNetEncoder, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv_stack = nn.ModuleList()
        self.conv_channels = [32, 64, 128, 128, 256, 256, 512]
        for i in range(6):
            stride = 1 if i % 2 else 2
            self.conv_stack.append(ConvBlock(self.conv_channels[i], self.conv_channels[i+1], stride=stride))

        for i in range(5):
            self.conv_stack.append(ConvBlock(512, 512, stride=1))

        self.conv_stack.append(ConvBlock(512, 1024, stride=2))
        self.conv_stack.append(ConvBlock(1024, 1024, stride=1))

    def forward(self, x):
        features = []
        x = self.initial_conv(x)
        features.append(x)
        for layer in self.conv_stack:
            x = layer(x)
            features.append(x)
        return features  # Return a list of feature maps for skip connections

class DepthDecoder(nn.Module): 
    def __init__(self, input_channels: int):
        super().__init__()
        
        self.iconv_1 = ConvBlock(input_channels=input_channels, output_channels=512, kernel_size=5, stride=1)
        self.iconv_2 = ConvBlock(input_channels=512, output_channels=256, kernel_size=5, stride=1)
        self.iconv_3 = ConvBlock(input_channels=512, output_channels=128, kernel_size=5, stride=1)
        self.iconv_4 = ConvBlock(input_channels=256, output_channels=64, kernel_size=5, stride=1)
        self.iconv_5 = ConvBlock(input_channels=96, output_channels=32, kernel_size=5, stride=1)
        
        self.output = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(1),
                                    nn.ReLU(inplace=True))
    
    def forward(self, features):
        x = self.iconv_1(features[-1])
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.iconv_2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.iconv_3(torch.cat([x, features[4]], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.iconv_4(torch.cat([x, features[2]], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.iconv_5(torch.cat([x, features[0]], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        x = self.output(x)
        return x
    
class FastDepth(nn.Module): 
    def __init__(self, input_channels: int): 
        super().__init__()
        
        self.encoder = MobileNetEncoder(input_channels=input_channels)
        self.decoder = DepthDecoder(input_channels=1024)
        
        if os.path.exists("MobileNet_Imagenet100.pth"):
            self.encoder.load_state_dict(torch.load("MobileNet_Imagenet100.pth", weights_only=True), strict=False)
        
    def forward(self, x): 
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    model = FastDepth(input_channels=3)
    input_tensor = torch.randn(1, 3, 224, 224)
    depth_output = model(input_tensor)
    print("Output shape:", depth_output.shape)
