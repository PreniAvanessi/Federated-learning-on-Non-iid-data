import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
      
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
      
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
  
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
     
        x = self.adaptive_pool(x)
        
     
        x = x.view(x.size(0), -1)
        
 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    