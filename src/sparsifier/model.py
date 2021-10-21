import torch
import os
import pudb
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from torch import nn

# https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/model.py


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),  # , inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),  # , inplace=True)
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
        # New arch v2
        self.lstm_num_layers = 2
        self.hidden_size = 128
        self.lstm1 = nn.LSTM(
            input_size=7296,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.lstm_num_layers,
            dropout=0.0,
        )
        self.conv1 = conv(True, 1024, 256, kernel_size=1, stride=1, dropout=0.0)
        self.conv2 = conv(True, 256, 128, kernel_size=3, stride=1, dropout=0.0)
        self.conv3 = conv(True, 128, 64, kernel_size=3, stride=1, dropout=0.0)

        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(422)
        self.fc3 = nn.Linear(1282, 1024)
        self.fc4 = nn.Linear(1024, 128)
        self.fc5 = nn.Linear(128, 2)
        # Pose
        self.fc3_pose = nn.Linear(1280, 1024)
        self.fc4_pose = nn.Linear(1024, 128)
        self.fc5_pose = nn.Linear(128, 2)

    def init_hidden(self, batch_size, hidden_size, device):
        # Just use default of all 0.
        h0 = np.zeros((self.lstm_num_layers, batch_size, hidden_size))
        h1 = np.zeros((self.lstm_num_layers, batch_size, hidden_size))
        h0 = torch.from_numpy(h0).float()
        h1 = torch.from_numpy(h1).float()
        h0 = h0.to(device)
        h1 = h1.to(device)
        return (h0, h1)

    def imageEncode(self, im):
        images = im.permute(1, 0, 2, 3, 4)
        ret = []
        for image in images:
            ret.append(self.encoder(image))
        ret = torch.stack(ret)
        ret = ret.permute(1, 0, 2, 3, 4)
        # Merge embedding dimension 512 * 7 * 7 -> 25088
        #        ret = ret.view(-1, ret.shape[1], ret.shape[2] * ret.shape[3] * ret.shape[4])
        return ret

    def encode(self, embedding):
        x = embedding.view(
            embedding.shape[0] * self.context,
            embedding.shape[2],
            embedding.shape[3],
            embedding.shape[4],
        )
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(
            embedding.shape[0], self.context, x.shape[1] * x.shape[2] * x.shape[3]
        )
        return x

    def forward(self, x1, x2):
        self.encoder.eval()
        out1 = self.imageEncode(x1)
        out2 = self.imageEncode(x2)
        x = torch.cat((out1, out2), 2)
        x = self.encode(x)
        x, hidden = self.lstm1(x, self.hidden)
        x = self.drop1(x)
        self.hidden = hidden
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x1 = self.fc3_pose(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.drop2(x1)
        x1 = self.fc4_pose(x1)
        x1 = F.relu(x1)
        pose = self.fc5_pose(x1)
        #        x1 = self.fc2(x)
        #        x1 = F.relu(x1)
        #        pose = self.fc3(x1)
        # Should we use embedding or prediction?
        # Prediction for now
        x = torch.cat((x, pose), 1)
        x = self.fc3(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop3(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return pose, x


class SiameseDeepVO(nn.Module):
    def __init__(self, context=10):
        super(SiameseDeepVO, self).__init__()
        self.context = 10
        self.hidden = None
        self.batchNorm = False
        # New arch v2
        self.device = None
        self.lstm_num_layers = 2
        self.hidden_size = 1000
        self.lstm1 = nn.LSTM(
            input_size=4096,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.lstm_num_layers,
            dropout=0.0,
        )
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3 = conv(
            self.batchNorm, 128, 256, kernel_size=5, stride=2, dropout=0.2
        )
        self.conv3_1 = conv(
            self.batchNorm, 256, 256, kernel_size=3, stride=1, dropout=0.2
        )
        self.conv4 = conv(
            self.batchNorm, 256, 512, kernel_size=3, stride=2, dropout=0.2
        )
        self.conv4_1 = conv(
            self.batchNorm, 512, 512, kernel_size=3, stride=1, dropout=0.2
        )
        self.conv5 = conv(
            self.batchNorm, 512, 512, kernel_size=3, stride=2, dropout=0.2
        )
        self.conv5_1 = conv(
            self.batchNorm, 512, 512, kernel_size=3, stride=1, dropout=0.2
        )
        self.conv6 = conv(
            self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=0.5
        )
        self.fc5 = nn.Linear(128, 2)
        # Pose
        self.rnn_drop_out = nn.Dropout(0.5)
        self.linear_out = 128
        self.fc1 = nn.Linear(self.hidden_size, self.linear_out)
        self.fc2 = nn.Linear(self.hidden_size + 128, self.linear_out)
        self.linearPose = nn.Linear(self.linear_out, 2)
        self.linearSim = nn.Linear(in_features=self.linear_out, out_features=2)

    def init_hidden(self, batch_size, hidden_size, device):
        # Just use default of all 0.
        h0 = np.zeros((self.lstm_num_layers, batch_size, hidden_size))
        h1 = np.zeros((self.lstm_num_layers, batch_size, hidden_size))
        h0 = torch.from_numpy(h0).float()
        h1 = torch.from_numpy(h1).float()
        h0 = h0.to(device)
        h1 = h1.to(device)
        return (h0, h1)

    #        images = im.permute(1, 0, 2, 3, 4)
    #        ret = []
    #        for image in images:
    #            ret.append(self.encoder(image))
    #        ret = torch.stack(ret)
    #        ret = ret.permute(1, 0, 2, 3, 4)
    #        # Merge embedding dimension 512 * 7 * 7 -> 25088
    #        return ret
    def imageEncode(self, im):
        batch_size = im.shape[0]
        ims = im.shape[1]
        im = im.view(batch_size * ims, im.shape[2], im.shape[3], im.shape[4])
        ret = self.encoder(im)
        ret = ret.view(batch_size, ims, -1)

        return ret

    def encoder(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

    def forward(self, x1, x2, truth_pose):
        x1 = torch.cat((x1[:, :-1], x1[:, 1:]), dim=2)
        x2 = torch.cat((x2[:, :-1], x2[:, 1:]), dim=2)
        x = self.imageEncode(torch.cat([x1, x2], dim=1))
        #        x = x.view(x.shape[0], x.shape[1], -1)
        x, hidden = self.lstm1(x, self.hidden)
        self.hidden = hidden
        #        x = x.reshape(x.shape[0], -1)
        x = x[:, -1, :]
        pose_embed = self.fc1(x)
        pose = F.relu(pose_embed)
        pose = self.linearPose(pose)
        #        T_GT = torch.norm(truth_pose[:, 0:3], dim=1)
        #        R_GT = truth_pose[:, 3]
        #        pose = torch.stack([T_GT, R_GT], 1)
        #        pose = torch.rand(pose.shape).to(self.device)

        x = torch.cat((x, pose_embed), 1)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.linearSim(x)
        return pose, x


if __name__ == "__main__":
    M_deepvo = SiameseDeepVO()
    pretrained = "./flownets_EPE1.951.pth.tar"
    pretrained_w = torch.load(pretrained)
    print("Load FlowNet pretrained model")
    # Use only conv-layer-part of FlowNet as CNN for DeepVO
    model_dict = M_deepvo.state_dict()
    update_dict = {
        k: v for k, v in pretrained_w["state_dict"].items() if k in model_dict
    }
    model_dict.update(update_dict)
    M_deepvo.load_state_dict(model_dict)
