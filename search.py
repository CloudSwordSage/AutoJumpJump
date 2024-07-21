import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SearchNet(nn.Module):
    def __init__(self, input_size=1024):
        super(SearchNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x