import torch
import torch.nn as nn
import torchvision.models as models
from utils import get_upsample_filter


class FCN8(nn.Module):
    def __init__(self, pretrain=True, output_size=1):
        super(FCN8, self).__init__()
        self.output_size = output_size

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.score_pool5 = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, output_size, 1),
        )

        self.score_pool4 = nn.Conv2d(512, output_size, 1)
        self.score_pool3 = nn.Conv2d(256, output_size, 1)

        self.upscore = nn.ConvTranspose2d(output_size, output_size, 16, stride=8)
        self.upscore4 = nn.ConvTranspose2d(output_size, output_size, 4, stride=2)
        self.upscore5 = nn.ConvTranspose2d(output_size, output_size, 4, stride=2)

        self.init_param(pretrain=pretrain)

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score5 = self.score_pool5(conv5)
        upscore5 = self.upscore5(score5)
        score4 = self.score_pool4(conv4)
        score4 = score4[:, :, 5:5 + upscore5.size()[2], 5:5 + upscore5.size()[3]].contiguous()
        score4 += upscore5

        upscore4 = self.upscore4(score4)
        score3 = self.score_pool3(conv3)
        score3 = score3[:, :, 9:9 + upscore4.size()[2], 9:9 + upscore4.size()[3]].contiguous()
        score3 += upscore4

        out = self.upscore(score3)
        out = out[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        return out

    def init_param(self, pretrain=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]
        if pretrain:
            ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
            vgg16 = models.vgg16(pretrained=True)
            features = list(vgg16.features.children())
            for idx, conv_block in enumerate(blocks):
                for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                    if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                        l2.weight.data = l1.weight.data
                        l2.bias.data = l1.bias.data
            for i1, i2 in zip([0, 3], [0, 3]):
                l1 = vgg16.classifier[i1]
                l2 = self.score_pool5[i2]
                l2.weight.data = l1.weight.data.view(l2.weight.size())
                l2.bias.data = l1.bias.data.view(l2.bias.size())
        else:
            for layer in blocks:
                for i in [0, 3, 6]:
                    torch.nn.init.kaiming_normal_(layer[i].weight.data)
                    torch.nn.init.constant_(layer[i].bias.data, val=0)
            for i in [0, 3]:
                torch.nn.init.kaiming_normal_(self.score_pool5[i].weight.data)
                torch.nn.init.constant_(self.score_pool5[i].bias.data, val=0)

        scores = [
            self.score_pool5[-1],
            self.score_pool4,
            self.score_pool3]
        for layer in scores:
            torch.nn.init.kaiming_normal_(layer.weight.data)
            torch.nn.init.constant_(layer.bias.data, val=0)

        upscore_layer = [
            self.upscore5,
            self.upscore4,
            self.upscore]
        for layer in upscore_layer:
            # initialize upscore layer
            c1, c2, h, w = layer.weight.data.size()
            assert c1 == c2 == self.output_size
            assert h == w
            weight = get_upsample_filter(h)
            layer.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)