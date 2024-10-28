import torch 
import torch.nn as nn 

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

class MobileNet(nn.Module): 
    def __init__(self, input_channels: int, num_of_class: int): 
        super(MobileNet, self).__init__() 

        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU()
        )
        
        self.conv_stack = nn.ModuleList()
        self.conv_channels = [32, 64, 128, 128, 256, 256, 512]
        for i in range(6): 
            stride = 1 if i % 2 else 2
            self.conv_stack.append(ConvBlock(self.conv_channels[i], self.conv_channels[i+1], stride))
        
        for i in range(5): 
            self.conv_stack.append(ConvBlock(512, 512, 1))
        
        self.conv_stack.append(ConvBlock(512, 1024, 2))
        self.conv_stack.append(ConvBlock(1024, 1024, 1))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_head = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(), 
            nn.Linear(512, num_of_class)
        )
        
    def forward(self, x): 
        x = self.initial_conv(x)
        for layer in self.conv_stack:
            x = layer(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_head(x)
        return x
    
def get_MobileNet(num_of_classes : int): 
    """
    Helper Function to get Mobilenet with defined parameters.
    """
    return MobileNet(3, num_of_class=num_of_classes).to("cuda" if torch.cuda.is_available() else "cpu")