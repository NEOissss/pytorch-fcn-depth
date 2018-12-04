import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, conv, writer, granularity=100):
        super(CombinedLoss, self).__init__()
        self.conv = conv
        self.granularity = granularity / 10
        self.writer = writer
        self.criterion1 = torch.nn.CrossEntropyLoss()
        self.criterion2 = torch.nn.MSELoss()

    def forward(self, x, gt):
        gt = gt * self.granularity
        int_gt = gt.type(torch.cuda.LongTensor)[0]
        loss1 = self.criterion1(x, int_gt)
        int_x = torch.unsqueeze(x.max(dim=1)[1], 1).type(torch.cuda.FloatTensor)
        est_x = self.conv(int_x)
        loss2 = self.criterion2(est_x, gt)
        self.writer.add_scalar('train_loss/CrossEntropy', loss1.item())
        self.writer.add_scalar('train_loss/MSE', loss2.item())
        return loss1 + 0.1 * loss2


class CombinedLoss_test(nn.Module):
    def __init__(self, conv, granularity=100):
        super(CombinedLoss_test, self).__init__()
        self.conv = conv
        self.granularity = granularity
        self.criterion = torch.nn.MSELoss()

    def forward(self, x, gt):
        gt = gt * self.granularity
        int_x = torch.unsqueeze(x.max(dim=1)[1], 1).type(torch.cuda.FloatTensor)
        est_x = self.conv(int_x)
        loss = self.criterion(est_x, gt)
        return loss
