import argparse
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from fcn8 import FCN8
from fcn2 import FCN2
from fcn2_spatial import FCN2_spatial
from dataset import NYUDv2Dataset
from loss import LogDepthLoss


class FCNManager(object):
    def __init__(self, net, data_opts, pretrain=True, param_path=None, lr=1e-3, decay=1e-2, batch=8, flip=False, val=True):
        self.batch = batch
        self.flip = flip
        self.batch = batch
        self.val = val
        self.granularity = 1
        self.writer = SummaryWriter()

        self._net = net
        if net == 'FCN8':
            self.net = torch.nn.DataParallel(FCN8(pretrain=pretrain, output_size=self.granularity)).cuda()
        elif net == 'FCN2':
            self.net = torch.nn.DataParallel(FCN2(pretrain=pretrain, output_size=self.granularity)).cuda()
        elif net == 'FCN2s':
            self.net = torch.nn.DataParallel(FCN2_spatial(pretrain=pretrain, output_size=self.granularity)).cuda()
        else:
            raise ValueError('Unknown model!')

        if param_path:
            self.load_param(param_path)

        # self.criterion = torch.nn.MSELoss().cuda()
        self.criterion = LogDepthLoss().cuda()
        self.solver = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.solver, verbose=True)

        self.data_opts = data_opts
        self.train_data_loader, self.test_data_loader, self.val_data_loader = self.data_loader()

    def train(self, epoch=1, lr=None):
        print('Training.')
        if lr:
            for g in self.solver.param_groups:
                g['lr'] = lr

        for t in range(epoch):
            iter_num = 0
            for data, depth in iter(self.train_data_loader):
                if self.flip:
                    idx = torch.randperm(data.size(0))[:data.size(0) // 2]
                    data[idx] = data[idx].flip(3)
                    depth[idx] = depth[idx].flip(3)
                self.solver.zero_grad()
                score = self.net(data)
                loss = self.criterion(score, depth)
                self.writer.add_scalar('train_loss/MSE', loss.item(), t*len(self.train_data_loader)+iter_num)
                loss.backward()
                self.solver.step()
                iter_num += 1

            if self.val:
                val_loss = self.test(val=True)
                self.writer.add_scalar('val_loss/MSE', val_loss.item(), t)
                self.scheduler.step(val_loss)

    def test(self, val=False, save_flag=False):
        if val:
            data_loader = self.val_data_loader
        else:
            data_loader = self.test_data_loader
            print('Testing.')

        self.net.eval()
        loss_list = []
        if save_flag:
            res_metric = np.zeros((len(data_loader), 480, 640))

        iter_num = 0
        for data, depth in iter(data_loader):
            score = self.net(data)
            loss = self.criterion(score, depth)
            loss_list.append(loss.item())
            if save_flag:
                res_metric[iter_num] = score.cpu().detach().squeeze().numpy()
            iter_num += 1

        if save_flag:
            np.save(self._net + '_loss', np.array(loss_list))
            np.save(self._net + '_res', res_metric)

        if not val:
            self.writer.add_scalar('test_loss/MSE', np.mean(loss_list).item())

        self.net.train()
        return np.mean(loss_list)

    def data_loader(self):
        if self.data_opts['dataset'] == 'NYUDv2':
            train_dataset = NYUDv2Dataset(self.data_opts['path'], self.data_opts['train_cut'])
            test_dataset = NYUDv2Dataset(self.data_opts['path'], self.data_opts['test_cut'])
            val_dataset = NYUDv2Dataset(self.data_opts['path'], self.data_opts['val_cut'])
            train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.batch, shuffle=True)
            test_data_loader = DataLoader(dataset=test_dataset, batch_size=self.batch)
            val_data_loader = DataLoader(dataset=val_dataset, batch_size=self.batch)
            return train_data_loader, test_data_loader, val_data_loader

    def save(self):
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        net_path = self._net + '-param-' + timestamp
        torch.save(self.net.state_dict(), net_path)
        print('FCN model parameters saved: ' + net_path)
        json_path = 'all_scalars' + timestamp + '.json'
        self.writer.export_scalars_to_json(json_path)
        print('Exp stats saved: ' + json_path)
        self.writer.close()

    def load_param(self, path):
        self.net.load_state_dict(torch.load(path))
        print('Net model parameters loaded: ' + path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--net', dest='net', type=str, default='FCN8', help='Network')
    parser.add_argument('--path', dest='path', type=str, default='./', help='Dataset path')
    parser.add_argument('--param', dest='param', type=str, default=None, help='Initial net parameters.')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Base learning rate for training.')
    parser.add_argument('--decay', dest='decay', type=float, default=0.01, help='Weight decay.')

    parser.add_argument('--batch', dest='batch', type=int, default=16, help='Batch size.')
    parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='Epochs for training.')
    parser.add_argument('--verbose', dest='verbose', type=int, default=1, help='Printing frequency setting.')

    parser.add_argument('--valid', dest='valid', action='store_true', help='Use validation.')
    parser.add_argument('--no-valid', dest='valid', action='store_false', help='Do not use validation.')
    parser.set_defaults(valid=True)

    args = parser.parse_args()

    if args.lr <= 0:
        raise AttributeError('--lr parameter must > 0.')
    if args.decay < 0:
        raise AttributeError('--decay parameter must >= 0.')
    if args.batch <= 0:
        raise AttributeError('--batch parameter must > 0.')
    if args.epoch <= 0:
        raise AttributeError('--epoch parameter must > 0.')

    data_opts = {'dataset': 'NYUDv2',
                 'path': args.path,
                 'train_cut': [0, 1000],
                 'test_cut': [1050, None],
                 'val_cut': [1000, 1050]}

    print('====' + args.net + ' for Depth Exp====')
    print('Validation: ' + str(args.valid))
    print('Net parameters: ' + str(args.param))
    print('#Epoch: {:d}, #Batch: {:d}'.format(args.epoch, args.batch))
    print('Learning rate: {:.0e}'.format(args.lr))
    print('Weight decay: {:.0e}'.format(args.decay))
    print('Learning rate scheduler used!\n')

    fcn = FCNManager(net=args.net, data_opts=data_opts, param_path=args.param, lr=args.lr, decay=args.decay, batch=args.batch, val=args.valid)
    fcn.train(epoch=args.epoch, lr=args.lr)
    fcn.test()
    fcn.save()


if __name__ == '__main__':
    main()
