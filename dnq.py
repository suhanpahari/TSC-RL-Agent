import torch.nn as nn
import cv2

class DQN(nn.Module):
    def __init__(self, d, a):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, a))
    
    def forward(self, x):
        return self.fc(self.conv(x))

def pre(f):
    f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    f = cv2.resize(f, (84, 84))
    return f / 255.0