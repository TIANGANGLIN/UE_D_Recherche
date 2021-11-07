"""
Test different methods, including OC-SVM, DeepSVDD, DROCC, PCA, kPCA
On MNIST, CIFAR10 dataset
"""
import os
import numpy as np
import pymp

# seed_value = 0
# torch.manual_seed(seed_value)
# random.seed(seed_value)
# np.random.seed(seed_value)


### test DeepSVDD
# %cd ./Ganglin/methods/Deep-SVDD-PyTorch/src
# python main.py cifar10 cifar10_LeNet ../log/cifar10_test ./Ganglin/data --load_model "./Ganglin/methods/Deep-SVDD-PyTorch/log/cifar10_test/model.tar" --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 256 --weight_decay 0.5e-6 --pretrain True --train True --test True --ae_lr 0.0001 --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 5

### test DROCC
# %cd ./Ganglin/methods/DROCC
# python  main_cifar.py  --lamda 1  --radius 8 --lr 0.001 --gamma 1 --ascent_step_size 0.001 --batch_size 256 --epochs 100 --optim 0 --normal_class 5 --eval 0

### test GANomaly
# %cd ./Ganglin/methods/ganomaly
# python train.py --dataset cifar10 --isize 32 --niter 150 --abnormal_class "dog" --manualseed 0

###############
# Train
###############
# training_cmds = [
#     "cd ./Ganglin/methods/Deep-SVDD-PyTorch/src && \
#     python main.py cifar10 cifar10_LeNet ../log/cifar10_test ./Ganglin/data --load_model './Ganglin/methods/Deep-SVDD-PyTorch/log/cifar10_test/model.tar' --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 256 --weight_decay 0.5e-6 --pretrain True --train True --test True --ae_lr 0.0001 --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 5",

#     "cd ./Ganglin/methods/DROCC && \
#     python  main_cifar.py  --lamda 1  --radius 8 --lr 0.001 --gamma 1 --ascent_step_size 0.001 --batch_size 256 --epochs 100 --optim 0 --normal_class 5 --eval 0",

#     "cd ./Ganglin/methods/ganomaly && \
#     python train.py --dataset cifar10 --isize 32 --niter 150 --abnormal_class 'dog' --manualseed 0 --eval 0"]

# num_thread = len(training_cmds)
# with pymp.Parallel(num_thread) as p:
#     for index in p.range(len(training_cmds)):
#         os.system(training_cmds[index])



###############
# Test
###############
# "cartoon_dog", "kaggle_testset", "cifra10"
testing_cmds = [
    
    "cd ./Ganglin/methods/Deep-SVDD-PyTorch/src && python main.py cifar10 cifar10_LeNet ../log/cifar10_test ./Ganglin/data --load_model './Ganglin/methods/Deep-SVDD-PyTorch/log/cifar10_test/model.tar' --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 256 --weight_decay 0.5e-6 --pretrain False --train False --test True --ae_lr 0.0001 --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 5 --dataset_name_test 'cifra10'",

    "cd ./Ganglin/methods/DROCC && python  main_cifar.py  --lamda 1  --radius 8 --lr 0.001 --gamma 1 --ascent_step_size 0.001 --batch_size 256 --epochs 100 --optim 0 --normal_class 5 --eval 1 --dataset_name 'cifra10'",

    "cd ./Ganglin/methods/ganomaly &&     python train.py --dataset cifar10 --isize 32 --niter 150 --abnormal_class 'dog' --manualseed 0 --eval 1  --dataset_name 'cifra10'"
]

num_thread = len(testing_cmds)
with pymp.Parallel(num_thread) as p:
    for index in p.range(len(testing_cmds)):
        os.system(testing_cmds[index])

