import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from torch import nn

#https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/model.py
def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)#, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)#, inplace=True)
        )

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

    def __init__(self, context=10):
        super(Siamese, self).__init__()
        self.context = 10
        self.hidden = None
        self.encoder = self.get_Resnet()
        #New arch v2
        self.lstm_num_layers = 2
        self.hidden_size = 128
        self.lstm1 = nn.LSTM(input_size=3136, hidden_size=self.hidden_size, batch_first=True, num_layers=self.lstm_num_layers, dropout=0.0)
        self.conv1 = conv(True, 1024, 256, kernel_size=1, stride=1, dropout=0.0)
        self.conv2 = conv(True, 256, 128, kernel_size=3, stride=1, dropout=0.0)
        self.conv3 = conv(True, 128, 64, kernel_size=3, stride=1, dropout=0.0)

        self.fc3 = nn.Linear(1282, 1024)
        self.fc4 = nn.Linear(1024, 128)
        self.fc5 = nn.Linear(128, 2)
        #Pose
        self.fc3_pose = nn.Linear(1280, 1024)
        self.fc4_pose = nn.Linear(1024, 128)
        self.fc5_pose = nn.Linear(128, 2)

    def init_hidden(self, batch_size, hidden_size, device):
        #Just use default of all 0.
        h0 = np.zeros((self.lstm_num_layers,batch_size,hidden_size))
        h1 = np.zeros((self.lstm_num_layers,batch_size,hidden_size))
        h0 = torch.from_numpy(h0).float()
        h1 = torch.from_numpy(h1).float()
        h0 = h0.to(device)
        h1 = h1.to(device)
        return (h0, h1)

    def imageEncode(self, im):
        images = im.permute(1,0,2,3,4)
        ret = []
        for image in images:
            ret.append(self.encoder(image))
        ret = torch.stack(ret)
        ret = ret.permute(1,0,2,3,4)
        #Merge embedding dimension 512 * 7 * 7 -> 25088
#        ret = ret.view(-1, ret.shape[1], ret.shape[2] * ret.shape[3] * ret.shape[4])
        return ret

    def encode(self,embedding):
        x = embedding.view(embedding.shape[0] * self.context, embedding.shape[2], embedding.shape[3], embedding.shape[4])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(embedding.shape[0], self.context, x.shape[1]*x.shape[2]*x.shape[3])
        return x

        

    def forward(self, x1, x2):
        self.encoder.eval()
        out1 = self.imageEncode(x1)
        out2 = self.imageEncode(x2)
        x = torch.cat((out1, out2), 2)
        x = self.encode(x)
        x, hidden = self.lstm1(x, self.hidden)
        self.hidden = hidden
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x1 = self.fc3_pose(x)
        x1 = F.relu(x1)
        x1 = self.fc4_pose(x1)
        x1 = F.relu(x1)
        pose = self.fc5_pose(x1)
#        x1 = self.fc2(x)
#        x1 = F.relu(x1)
#        pose = self.fc3(x1)
        #Should we use embedding or prediction?
        #Prediction for now
        x = torch.cat( (x,pose), 1)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return pose, x

