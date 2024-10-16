import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(CNNActionValue, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)  # [N, 4, 84, 84] -> [N, 16, 20, 20]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # [N, 16, 20, 20] -> [N, 32, 9, 9]
        self.in_features = 32 * 9 * 9
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)
        self.activation = activation

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ImprovedCNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(ImprovedCNNActionValue, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=8, stride=4)  # [N, state_dim, 84, 84] -> [N, 32, 20, 20]
        self.ln1 = nn.LayerNorm([32, 20, 20])
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # [N, 32, 20, 20] -> [N, 64, 9, 9]
        self.ln2 = nn.LayerNorm([64, 9, 9])
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)  # [N, 64, 9, 9] -> [N, 128, 7, 7]
        self.ln3 = nn.LayerNorm([128, 7, 7])

        self.in_features = 128 * 7 * 7  # 마지막 Conv 출력의 feature size
        
        self.fc1 = nn.Linear(self.in_features, 512)
        self.ln4 = nn.LayerNorm(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.ln5 = nn.LayerNorm(256)
        
        self.fc3 = nn.Linear(256, action_dim)

        self.activation = activation

    def forward(self, x):
        # Conv layers with LayerNorm and activation
        x = self.activation(self.ln1(self.conv1(x)))
        x = self.activation(self.ln2(self.conv2(x)))
        x = self.activation(self.ln3(self.conv3(x)))
        
        # Flatten the output from conv layers
        x = x.view(-1, self.in_features)
        
        # Fully connected layers with LayerNorm and activation
        x = self.activation(self.ln4(self.fc1(x)))
        x = self.activation(self.ln5(self.fc2(x)))
        
        # Output layer (no activation, since it's the output for action values)
        x = self.fc3(x)
        
        return x


