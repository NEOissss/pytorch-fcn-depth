#!/bin/sh

path='nyu_depth_v2_labeled.mat'

param=''
#param='--n_param FCN8s-param-467382378'

lr='--lr 0.001'
decay='--decay 0.01'
epoch='--epoch 10'
batch='--batch 128'
verbose='--verbose 1'

valid='--valid'
#valid='--no-valid'

python3 fcn.py ${path} ${param} ${lr} ${decay} ${epoch} ${batch} ${verbose} ${valid}
