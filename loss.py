import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, conv):
        super(CombinedLoss, self).__init__()
        self.conv = conv
        self.criterion1 = torch.nn.CrossEntropyLoss()
        self.criterion2 = torch.nn.MSELoss()

    def forward(self, x, gt):
        int_gt = gt.type(torch.LongTensor).cuda()
        loss1 = self.criterion1(x, int_gt)
        int_x = x.max(dim=1)[1]
        est_x = self.conv(int_x)
        loss2 = self.criterion2(est_x, gt)
        return loss1 + loss2