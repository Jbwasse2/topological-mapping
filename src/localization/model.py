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
        # New arch, RGB stuff
        self.encoder = self.get_Resnet()
        self.conv1 = nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(62720, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, 1)
        # Depth stuff
        self.convd1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=3)
        self.convd2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=3
        )
        self.convd3 = nn.Conv2d(
            in_channels=32, out_channels=128, kernel_size=5, stride=3
        )
        self.convd4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.convd5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.convd6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        # NEW
        self.convd1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=10, stride=4
        )
        self.convd2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=3
        )
        self.convd3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2
        )
        self.convd4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1
        )
        self.convd5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)

    def encode(self, im):
        x = self.encoder(im)
        return x

    def encode_depth(self, im):
        x = self.convd1(im)
        x = F.relu(x)
        x = self.convd2(x)
        x = F.relu(x)
        x = self.convd3(x)
        x = F.relu(x)
        x = self.convd4(x)
        x = F.relu(x)
        x = self.convd5(x)
        x = F.relu(x)
        return x

    def forward(self, x1, x2,d1,d2):
        # Get embedding for RGB
        self.encoder.eval()
        out1 = self.encode(x1)
        out2 = self.encode(x2)
        xc = torch.cat((out1, out2), 1)
        # Get embedding for depth
        out1 = self.encode_depth(d1)
        out2 = self.encode_depth(d2)
        xd = torch.cat((out1, out2), 1)
        # Combine depth a RGB embeddings and forawrd pass through model
        x = torch.cat((xc, xd), 1)
#        x = self.conv1(x)
#        x = F.relu(x)
#        x = self.conv2(x)
#        x = F.relu(x)
#        x = self.conv3(x)
#        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

    # def forward(self, x1, x2, d1, d2):
    #    # Get embedding for RGB
    #    self.encoder.eval()
    #    out1 = self.encode(x1)
    #    out2 = self.encode(x2)
    #    out = torch.cat((out1, out2), 1)
    #    xc = self.conv1(out)
    #    xc = F.relu(xc)
    #    xc = self.conv2(xc)
    #    xc = F.relu(xc)
    #    xc = self.conv3(xc)
    #    xc = F.relu(xc)
    #    xc = xc.view(xc.size(0), -1)
    #    # Get embedding for depth
    #    out1 = self.encode_depth(d1)
    #    out2 = self.encode_depth(d2)
    #    xd = torch.cat((out1, out2), 1)
    #    xd = self.conv1(out)
    #    xd = F.relu(xd)
    #    xd = self.conv2(xd)
    #    xd = F.relu(xd)
    #    xd = self.conv3(xd)
    #    xd = F.relu(xd)
    #    xd = xd.view(xd.size(0), -1)
    #    # Combine depth a RGB embeddings and forawrd pass through model
    #    x = torch.cat((xc, xd), 1)
    #    x = self.fc1(x)
    #    x = F.relu(x)
    #    x = self.dropout1(x)
    #    x = self.fc2(x)
    #    return x

class SiameseRGB(nn.Module):
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
        super(SiameseRGB, self).__init__()
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
        #New arch
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc1_5 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def encode(self, im):
        x = self.encoder(im)
        #x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        return x

    def forward(self, x1, x2):
        #x1 = nnf.interpolate(x1, size=(128,128))
        #x2 = nnf.interpolate(x2, size=(128,128))
        self.encoder.eval()
        out1 = self.encode(x1)
        out2 = self.encode(x2)
        x = torch.cat((out1, out2), 1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc1_5(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
