import torch
import torch.nn.functional as nnf
import torch.nn.functional as F
import torchvision.models as models
from torch import nn


class Siamese(nn.Module):
    def get_Resnet(self):
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:-2]
        resnet18 = nn.Sequential(*modules)
        for param in resnet18.parameters():
            param.requires_grad = False
        # This was gotten by inspecting resnet18
        model = nn.Sequential(
            resnet18,
        )
        return model

    def __init__(self):
        super(Siamese, self).__init__()
        self.encoder = self.get_Resnet()
        self.conv1 = nn.Conv2d(512, 32, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.relu0 = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.out1 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)
        self.out2 = nn.Linear(128, 2)
        self.batch1 = nn.BatchNorm2d(32)
        self.batch2 = nn.BatchNorm1d(256)
        self.batch3 = nn.BatchNorm1d(128)
        self.batch4 = nn.BatchNorm1d(128)
        # New arch
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 2)

    def encode(self, im):
        x = self.encoder(im)
        # x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        return x

    def forward(self, x1, x2):
        # x1 = nnf.interpolate(x1, size=(128,128))
        # x2 = nnf.interpolate(x2, size=(128,128))
        self.encoder.eval()
        out1 = self.encode(x1)
        out2 = self.encode(x2)
        out = torch.cat((out1, out2), 1)
        x = self.conv1(out)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
