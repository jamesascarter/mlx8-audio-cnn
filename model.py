import torch
import torch.nn as nn
import torch.nn.functional as F

DROPOUT_RATE = 0.3

class AudioCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) # 1 channel input, 32 filters, 3x3 kernel, stride 1, padding 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 32 filters, 64 filters, 3x3 kernel, stride 1, padding 1
        self.dropout1 = nn.Dropout2d(DROPOUT_RATE)  # Spatial dropout
        self.dropout2 = nn.Dropout(DROPOUT_RATE)
        self.fc1 = nn.Linear(64 * 16 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.bn1 = nn.BatchNorm2d(32) 
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.dropout1(x) 
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 16 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x