"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import Ganomaly
import torchvision
from torchvision import transforms
import sys
sys.path.insert(0,"./Ganglin/methods/DROCC")
from data_process_scripts.process_cifar import CIFAR10_Dataset

##
def train():
    """ Training
    """
    

    ##
    # ARGUMENTS
    opt = Options().parse()
    ##
    # LOAD DATA
    # dataloader = load_data(opt)
    normalize_k = transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
    transform_k  = transforms.Compose([
            #  transforms.RandomResizedCrop(224),
            transforms.Resize([32,32]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_k,])
            
    normal_class = 5
    dataset = CIFAR10_Dataset("./Ganglin/data", normal_class)
    train_loader, test_loader = dataset.loaders(batch_size=opt.batchsize, num_workers=opt.workers)
    dataloader = {}
    dataloader['train'] = train_loader

    if opt.dataset_name == "cifra10":
        dataloader['test'] = test_loader
        dataset_k = None
    elif opt.dataset_name == "kaggle_testset":
        dataset_k = torchvision.datasets.ImageFolder(root='./Ganglin/data/kaggle_testset',transform=transform_k)
        kaggle_test_loader = torch.utils.data.DataLoader(
                                dataset_k,
                                batch_size=256,
                                shuffle=True)
        dataloader['test'] = kaggle_test_loader
    else:
        dataset_k = torchvision.datasets.ImageFolder(root='./Ganglin/data/cartoon_dog',transform=transform_k)
        kaggle_test_loader = torch.utils.data.DataLoader(
                                dataset_k,
                                batch_size=256,
                                shuffle=True)
        dataloader['test'] = kaggle_test_loader
    
    
    ##
    # LOAD MODEL
    model = Ganomaly(opt, dataloader)
    ##
    
    if not opt.eval:
        # TRAIN MODEL
        model.train()
    else:
        # Test model
        model.test(dataset_k)

if __name__ == '__main__':
    import torch
    import numpy as np
    import random

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(20)
    train()
