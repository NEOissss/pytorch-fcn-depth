import argparse
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from fcn import FCN8s
from loss import CombinedLoss, CombinedLoss_test
from dataset import NYUDv2Dataset


class FCNManager(object):
    def __init__(self, data_opts, pretrain=True, param_path=None, lr=1e-3, decay=1e-2, batch=8, flip=False, val=True):
        self.batch = batch
        self.flip = flip
        self.batch = batch
        self.val = val
        self.granularity = 100
        self.writer = SummaryWriter()

        self.net = torch.nn.DataParallel(FCN8s(pretrain=pretrain, output_size=self.granularity)).cuda()
        if param_path:
            self.load_param(param_path)
        # self.criterion = torch.nn.MSELoss().cuda()
        self.criterion = CombinedLoss(self.net.module.final_score, self.writer, self.granularity).cuda()
        self.criterion_test = CombinedLoss_test(self.net.module.final_score, self.granularity).cuda()
        self.solver = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.solver, verbose=True, patience=5)

        self.data_opts = data_opts
        self.train_data_loader, self.test_data_loader, self.val_data_loader = self.data_loader()
        self.loss_stats = []

    def train(self, epoch=1, verbose=None):
        print('Training.')
        self.loss_stats = []
        for t in range(epoch):
            # print("\nEpoch: {:d}".format(t + 1))
            iter_num = 0
            for data, depth in iter(self.train_data_loader):
                # print('Batch: {:d}'.format(iter_num), end=' ')
                if self.flip:
                    idx = torch.randperm(data.size(0))[:data.size(0) // 2]
                    data[idx] = data[idx].flip(3)
                    depth[idx] = depth[idx].flip(3)
                self.solver.zero_grad()
                score = self.net(data)
                # print('Training loss:', end=' ')
                loss = self.criterion(score, depth)
                self.loss_stats.append(loss.item())
                loss.backward()
                self.solver.step()
                iter_num += 1

                if self.val:
                    val_loss = self.test(val=True)
                    self.writer.add_scalar('val_loss/MSE', val_loss.item())

                # if self.val and verbose and iter_num % verbose == 0:
                #     print('Validation MSELoss: {:.2f}'.format(val_loss.item()))
                # else:
                #     print()

            if self.val:
                self.scheduler.step(val_loss)

        self.loss_stats = np.array(self.loss_stats)
        self.save()

    def test(self, val=False):
        if val:
            data_loader = self.val_data_loader
        else:
            data_loader = self.test_data_loader
            print('Testing.')

        self.net.eval()
        loss_list = []

        for data, depth in iter(data_loader):
            score = self.net(data)
            loss = self.criterion_test(score, depth)
            loss_list.append(loss.item())

        if not val:
            self.writer.add_scalar('test_loss/MSE', np.mean(loss_list).item())
            # print('Test MSE loss: {:f}'.format(np.mean(loss_list)))

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
        net_path = 'FCN8s-param-' + timestamp
        torch.save(self.net.state_dict(), net_path)
        print('FCN model parameters saved: ' + net_path)
        # stats_path = 'train_stats_' + timestamp
        # np.save(stats_path, self.loss_stats)
        # print('Training stats saved: ' + stats_path + '.npy\n')
        self.writer.export_scalars_to_json('all_scalars' + timestamp + '.json')
        self.writer.close()

    def load_param(self, path):
        self.net.load_state_dict(torch.load(path))
        print('Net model parameters loaded: ' + path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', dest='path', type=str, default='nyu_depth_v2_labeled.mat', help='Dataset path')
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
                 'train_cut': [0, 800],
                 'test_cut': [900, None],
                 'val_cut': [800, 900]}

    print('====FCN8s for Depth Exp====')
    print('Validation: ' + str(args.valid))
    print('Net parameters: ' + str(args.param))
    print('#Epoch: {:d}, #Batch: {:d}'.format(args.epoch, args.batch))
    print('Learning rate: {:.0e}'.format(args.lr))
    print('Weight decay: {:.0e}'.format(args.decay))
    print('Learning rate scheduler used!\n')

    fcn = FCNManager(data_opts=data_opts, param_path=args.param, lr=args.lr, decay=args.decay, batch=args.batch, val=args.valid)
    fcn.train(epoch=args.epoch, verbose=args.verbose)
    fcn.test()


if __name__ == '__main__':
    main()
