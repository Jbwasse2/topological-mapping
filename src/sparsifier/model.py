import torch
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
        self.linear = nn.Sequential(nn.Linear(25088, 4096), nn.Sigmoid())
        self.conv1 = nn.Conv2d(512, 32, 4)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.relu0 = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.out1 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.5)
        self.out2 = nn.Linear(128, 2)
        # self.batch1 = nn.BatchNorm2d(32)
        # self.batch2 = nn.BatchNorm1d(256)
        # self.batch3 = nn.BatchNorm1d(128)
        # self.batch4 = nn.BatchNorm1d(128)

    def encode(self, im):
        x = self.encoder(im)
        x = self.conv1(x)
        x = self.relu0(x)
        # x = self.batch1(x)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        # x = self.batch2(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop1(x)
        # x = self.batch3(x)
        return x

    def forward(self, x1, x2):
        out1 = self.encode(x1)
        out2 = self.encode(x2)
        out = torch.cat((out1, out2), 1)
        out = self.out1(out)
        out = self.relu3(out)
        out = self.drop3(out)
        # out = self.batch4(out)
        out = self.out2(out)
        return out
