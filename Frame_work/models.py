import torch
import torch.nn as nn
import torch.nn.functional as F
class NaturalSceneClassificationBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(262144,128)
        )
    
    def forward(self, xb):
        return self.network(xb)
    
class Model1Classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(128,9),
            nn.Softmax(dim=1)
        )
    
    def forward(self, xb):
        return self.network(xb)
    
    
class Model2Classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(128,32),
            nn.Softmax(dim=1)
        )
    
    def forward(self, xb):
        return self.network(xb)
    
    
h = 4
x = 3
y = 256
z = 256
random_image = torch.randn(h, x, y, z)
a = NaturalSceneClassificationBase()
b = Model2Classification()
print(b.forward(a.forward(random_image)).shape)