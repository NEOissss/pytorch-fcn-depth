import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg16


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class FCN(nn.Module):
    def __init__(self, pretrain=True, image_size=(480, 640), output_size=100):
        super(FCN, self).__init__()
        self.image_size = image_size
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

        self.score_pool5 = nn.Conv2d(512, output_size, 1)
        self.score_pool4 = nn.Conv2d(512, output_size, 1)
        self.score_pool3 = nn.Conv2d(256, output_size, 1)
        self.score_pool2 = nn.Conv2d(128, output_size, 1)
        self.score_pool1 = nn.Conv2d(64, output_size, 1)

        self.upscore5 = nn.ConvTranspose2d(output_size, output_size, 4, stride=2)
        self.upscore4 = nn.ConvTranspose2d(output_size, output_size, 4, stride=2)
        self.upscore3 = nn.ConvTranspose2d(output_size, output_size, 4, stride=2)
        self.upscore2 = nn.ConvTranspose2d(output_size, output_size, 4, stride=2)
        self.upscore1 = nn.ConvTranspose2d(output_size, output_size, 4, stride=2)

        self.fc_score_pool5 = nn.Sequential(
            nn.Linear(22 * 27, 22 * 27),
            nn.ReLU(inplace=True),
            nn.Linear(22 * 27, 22 * 27)
        )
        self.fc_score_pool4 = nn.Sequential(
            nn.Linear(43 * 53, 43 * 53),
            nn.ReLU(inplace=True),
            nn.Linear(43 * 53, 43 * 53)
        )

        self.final_score = nn.Sequential(
            nn.Conv2d(output_size, 64, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

        self.init_params(pretrain=pretrain)

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score5 = self.score_pool5(conv5)
        score5 = score5.view(1, -1)
        score5 = self.fc_score_pool5(score5)
        score5 = score5.view(1, 1, 22, 27)
        upscore5 = self.upscore5(score5)

        score4 = self.score_pool4(conv4)
        score4 = score4.view(1, -1)
        score4 = self.fc_score_pool4(score4)
        score4 = score4.view(1, 1, 43, 53)
        score4 += upscore5[:, :, 1:1+score4.size(2), 1:1+score4.size(3)]
        upscore4 = self.upscore4(score4)

        score3 = self.score_pool3(conv3)
        score3 += upscore4[:, :, 1:1+score3.size(2), 1:1+score3.size(3)]
        upscore3 = self.upscore3(score3)

        score2 = self.score_pool2(conv2)
        score2 += upscore3[:, :, 1:1+score2.size(2), 1:1+score2.size(3)]
        upscore2 = self.upscore2(score2)

        score1 = self.score_pool1(conv1)
        score1 += upscore2[:, :, 1:1+score1.size(2), 1:1+score1.size(3)]
        upscore1 = self.upscore1(score1)

        out = upscore1[:, :, 100:100+x.size()[2], 100:100+x.size()[3]].contiguous()
        out = self.final_score(out)

        return out

    def init_params(self, pretrain=True):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]
        if pretrain:
            ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
            features = list(vgg16(pretrained=True).features.children())
            for idx, conv_block in enumerate(blocks):
                for l1, l2 in zip(features[ranges[idx][0]: ranges[idx][1]], conv_block):
                    if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                        assert l1.weight.size() == l2.weight.size()
                        assert l1.bias.size() == l2.bias.size()
                        l2.weight.data = l1.weight.data
                        l2.bias.data = l1.bias.data
            # for i1, i2 in zip([0, 3], [0, 3]):
            #     l1 = vgg16(pretrained=True).classifier[i1]
            #     l2 = self.score_pool5[i2]
            #     l2.weight.data = l1.weight.data.view(l2.weight.size())
            #     l2.bias.data = l1.bias.data.view(l2.bias.size())
            # l1 = vgg16(pretrained=True).classifier[6]
            # l2 = self.score_pool5[6]
            # l2.weight.data = l1.weight.data[:self.output_size, :].view(l2.weight.size())
            # l2.bias.data = l1.bias.data[:self.output_size]
        else:
            for layer in blocks:
                for i in [0, 3, 6]:
                    torch.nn.init.kaiming_normal_(layer[i].weight.data)
                    torch.nn.init.constant_(layer[i].bias.data, val=0)

        # initialize score pool layers
        fc_scores = [
            self.fc_score_pool5,
            self.fc_score_pool4
        ]
        for layer in fc_scores:
            for i in [0, 2]:
                torch.nn.init.kaiming_normal_(layer[i].weight.data)
                torch.nn.init.constant_(layer[i].bias.data, val=0)

        scores = [
            self.score_pool5,
            self.score_pool4,
            self.score_pool3,
            self.score_pool2,
            self.score_pool1
            ]
        for layer in scores:
            torch.nn.init.kaiming_normal_(layer.weight.data)
            torch.nn.init.constant_(layer.bias.data, val=0)

        for i in [0, 2, 4, 6]:
            torch.nn.init.kaiming_normal_(self.final_score[i].weight.data)
            torch.nn.init.constant_(self.final_score[i].bias.data, val=0)

        # initialize upscore layers
        upscore_layer = [
            self.upscore5,
            self.upscore4,
            self.upscore3,
            self.upscore2,
            self.upscore1
        ]
        for layer in upscore_layer:
            c1, c2, h, w = layer.weight.data.size()
            assert c1 == c2 == self.output_size
            assert h == w
            weight = get_upsample_filter(h)
            layer.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
