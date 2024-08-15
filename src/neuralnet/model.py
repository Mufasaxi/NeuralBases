import torch.nn as nn

class EndgameModel(nn.Module):
    
    def __init__(self):
        """
        Defines the architecture for the Neural Network

        conv2d  ->  conv2d  -> conv2d  ->  pool  ->  fc  -> fc  ->  ReLU
        """
        super(EndgameModel, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(256 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the layers of the model
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x