import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super(ConvBlock, self).__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=stride, padding=1, groups=input_channels),
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

class MobileNetEncoder(nn.Module):
    def __init__(self, input_channels: int):
        super(MobileNetEncoder, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv_stack = nn.ModuleList()
        self.conv_channels = [64, 64, 128, 128, 256, 256, 512]
        for i in range(6):
            stride = 1 if i % 2 else 2
            self.conv_stack.append(ConvBlock(self.conv_channels[i], self.conv_channels[i+1], stride))

        for i in range(5):
            self.conv_stack.append(ConvBlock(512, 512, 1))

        self.conv_stack.append(ConvBlock(512, 1024, 2))
        self.conv_stack.append(ConvBlock(1024, 1024, 1))

    def forward(self, x):
        features = []
        x = self.initial_conv(x)
        features.append(x)
        for layer in self.conv_stack:
            x = layer(x)
            features.append(x)
        return features  # Return a list of feature maps for skip connections

class Decoder(nn.Module): 
    def __init__(self, input_channels: int):
        super().__init__()
        self.upconv5 = nn.ConvTranspose2d(input_channels, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.iconv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.iconv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.iconv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.iconv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.iconv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.depth_predictor = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
    
    def forward(self, features):
        x = features[-1]
        x = self.upconv5(x)
        
        x = self.upconv4(x)
        x = x + features[4]
        x = self.iconv4(x)
        
        x = self.upconv3(x)
        x = x + features[2]
        x = self.iconv3(x)
        
        x = self.upconv2(x)
        x = x + features[0]
        x = self.iconv2(x)
        
        x = self.upconv1(x)
        x = self.iconv1(x)
        
        depth = self.depth_predictor(x)
        depth = torch.sigmoid(depth)

        return depth

# Example usage
if __name__ == "__main__":
    encoder = MobileNetEncoder(input_channels=3)
    decoder = Decoder(input_channels=1024)
    input_tensor = torch.randn(1, 3, 224, 224)  # Example input
    features = encoder(input_tensor)
    for feature in features:
        print("Feature shape:", feature.shape)  # Print shape of each feature map
    depth_output = decoder(features)
    print("Output shape:", depth_output.shape)  # Expected output shape: (1, 1, 256, 256)
