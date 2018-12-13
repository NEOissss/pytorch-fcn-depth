from trainer import FCNManager


def main():
    net = ['FCN8', 'FCN2', 'FCN2s']
    param = ['FCN8-param-20181211130455', 'FCN2-param-20181211191402', 'FCN2s-param-20181212012621']

    data_opts = {'dataset': 'NYUDv2',
                 'path': '../dataset/NYUDepthV2/',
                 'train_cut': [0, 1000],
                 'test_cut': [0, 1000],
                 'val_cut': [0, 1000]}

    for i in range(3):
        fcn = FCNManager(net=net[i], data_opts=data_opts, param_path=param[i], batch=1)
        print(net[i] + ' Loss: {:f}'.format(fcn.test(val=True)))

    data_opts = {'dataset': 'NYUDv2',
                 'path': '../dataset/NYUDepthV2/',
                 'train_cut': [0, 1000],
                 'test_cut': [0, 1000],
                 'val_cut': [1200, 1210]}

    for i in range(3):
        fcn = FCNManager(net=net[i], data_opts=data_opts, param_path=param[i], batch=1)
        print(net[i] + ' Loss: {:f}'.format(fcn.test(val=True, save_idx=[i for i in range(10)])))


if __name__ == '__main__':
    main()