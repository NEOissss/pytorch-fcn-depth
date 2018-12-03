#!/bin/sh

path='--path data/'

param=''
#param='--n_param FCN8s-param-467382378'

lr='--lr 0.001'
decay='--decay 0.01'
epoch='--epoch 1'
batch='--batch 32'
verbose='--verbose 1'

valid='--valid'
#valid='--no-valid'

python3 trainer.py ${path} ${param} ${lr} ${decay} ${epoch} ${batch} ${verbose} ${valid}
