import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvARG(nn.Module):
    def __init__(self, inchannel):
        super().__init__()
        self.conv1 = nn.Conv1d(inchannel, 32, 39,padding=19)
        self.conv40 = nn.Conv1d(32,32,39,padding=19)
        self.pool1 = nn.MaxPool1d(5, 2)
        self.conv2 = nn.Conv1d(32, 64, 29,padding=14)
        self.conv3 = nn.Conv1d(64, 128, 29,padding=14)
        self.conv30 = nn.Conv1d(64,64,29,padding =14)
        self.conv30_128 = nn.Conv1d(128,128,29,padding=14)
        self.pool2 = nn.MaxPool1d(5, 2)
        self.conv4 = nn.Conv1d(128, 256, 19, padding=9)
        self.conv5 = nn.Conv1d(256, 256, 19, padding=9)
        self.conv20 = nn.Conv1d(256,256,19, padding=9)
        self.pool3 = nn.MaxPool1d(4, 1)
        self.conv6 = nn.Conv1d(256, 256, 19, padding=9)
        self.pool4 = nn.MaxPool1d(2,1)
        self.fc1 = nn.Linear(49664, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

        self.dim32to64 = nn.Conv1d(32, 64, 1)
        self.dim64to128 = nn.Conv1d(64, 128, 1)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        # x = self.pool1(x)
        # print(x.shape)
        x = F.relu(self.conv40(x)) + x
        x = F.relu(self.conv40(x)) + x
        x = F.relu(self.conv40(x)) + x
        x = self.pool1(x)

        x = F.relu(self.conv2(x)) + self.dim32to64(x)
        x = F.relu(self.conv30(x)) + x
        x = F.relu(self.conv30(x)) + x
        x = F.relu(self.conv30(x)) + x
        x = self.pool2(x)

        x = F.relu(self.conv3(x)) + self.dim64to128(x)
        x = F.relu(self.conv30_128(x)) + x
        x = F.relu(self.conv30_128(x)) + x
        x = F.relu(self.conv30_128(x)) + x
        x = self.pool3(x)
        
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv20(x)) + x
        # x = F.relu(self.conv20(x)) + x
        # # x = F.relu(self.conv20(x)) + x
        # x = self.pool4(x)
        
        # x = self.pool3(F.relu(self.conv5(F.relu(self.conv4(x)))))
        # x = self.pool4(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.shape)
        m = nn.Dropout(p=0.9)
        x = F.relu(self.fc1(x))
        x=m(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        m = nn.Softmax(dim=1)
        x = m(x)
        return x

net = ConvARG(23)
