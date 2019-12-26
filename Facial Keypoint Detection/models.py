## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1,16,3, padding=1)
        nn.init.xavier_uniform(self.conv1.weight)
        self.conv2 = nn.Conv2d(16,32,3, padding=1)
        nn.init.xavier_uniform(self.conv2.weight)
        self.conv3 = nn.Conv2d(32,64,3, padding=1)
        nn.init.xavier_uniform(self.conv3.weight)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(50176,300)
        self.fc2 = nn.Linear(300,136)
        self.fc1_drop = nn.Dropout(p=0.1)
        self.fc2_drop = nn.Dropout(p=0.2)        
        self.batch_norm = nn.BatchNorm1d(num_features=300)
        

        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 50176)
        x = self.fc1_drop(x)
        x = F.relu(self.batch_norm(self.fc1(x)))
        x = self.fc2_drop(x)
        x = self.fc2(x)

        return x
