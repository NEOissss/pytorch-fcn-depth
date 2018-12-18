import torch
import torch.nn as nn


class Discrete_CELoss(nn.Module):
    def __init__(self, granularity):
        super(Discrete_CELoss, self).__init__()
        self.granularity = granularity / 10
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, gt):
        gt = gt * self.granularity
        int_gt = gt.type(torch.cuda.LongTensor)[0]
        return self.criterion(x, int_gt)


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
        # int_x = torch.unsqueeze(x.max(dim=1)[1], 1).type(torch.cuda.FloatTensor)
        est_x = self.conv(x)
        loss2 = self.criterion2(est_x, gt)
        self.writer.add_scalar('train_loss/CE', loss1.item())
        self.writer.add_scalar('train_loss/MSE', loss2.item())
        return loss1 + 0.1 * loss2


class LogDepthLoss(nn.Module):
    def __init__(self):
        super(LogDepthLoss, self).__init__()

    def forward(self, x, gt):
        N = x.size(2) * x.size(3)
        d = x.log() - gt.log()
        ddx = torch.zeros(x.size()).cuda()
        ddy = torch.zeros(x.size()).cuda()
        ddx[:,:,:-1,:] = d[:,:,1:,:] - d[:,:,:-1,:]
        ddy[:,:,:,:-1] = d[:,:,:,1:] - d[:,:,:,:-1]
        loss = 1/N**2 * torch.sum(d**2, dim=(2, 3)) - 1/(2*N**2) * d.sum(dim=(2,3))**2 + 1/N * (ddx**2 + ddy**2).sum(dim=(2,3))
        return loss.mean()
