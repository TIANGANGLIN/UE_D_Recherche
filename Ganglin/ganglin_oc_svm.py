# load torch
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
from datasets.preprocessing import get_target_label_idx, global_contrast_normalization
import numpy as np
# other utilities
import argparse
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
# own package
from ganglin_MNIST import Novelty_MNIST
from ganglin_CIFAR10 import Novelty_CIFAR10


"""
The different between OS-SVM and SVDD
"""
for i in range(10):
    normal_class = i
    ##############################################
    # mnist
    ##############################################
    # mnist = Novelty_MNIST(normal_class=normal_class)
    # train_data, train_label = mnist.MNIST_DATASET_TRAIN()
    # test_data, test_label   = mnist.MNIST_DATASET_TEST()
    # training_features       = train_data.reshape(-1,28*28)
    # test_features           = test_data.reshape(-1,28*28)

    ##############################################
    # cifra10 dataset  
    ##############################################
    cifra10 = Novelty_CIFAR10(normal_class=normal_class)
    train_data, train_label = cifra10.DATASET_TRAIN()
    test_data, test_label   = cifra10.DATASET_TEST()
    training_features       = train_data.reshape(-1,32*32*3)
    test_features           = test_data.reshape(-1,32*32*3)

    # Training SVM
    print('------Training and testing SVM------')
    print("normal_class = ",normal_class)
    clf = svm.OneClassSVM(kernel="rbf",nu=0.05) # kernel = "sigmoid"
    clf.fit(training_features)
    
    #Test on Training data
    train_result = clf.predict(training_features)
    precision = sum(train_result == train_label)/train_label.shape[0]
    print('Training precision: {:.4f}'.format(precision))
    # AUC = roc_auc_score(train_label,train_result)
    # print('AUC = {:.4f}'.format(AUC))

    #Test on test data
    test_result = clf.predict(test_features)
    precision = sum(test_result == test_label)/test_label.shape[0]
    AUC = roc_auc_score(test_label,test_result)
    print('Test precision: {:.4f}'.format(precision))
    print('AUC = {:.4f}'.format(AUC))
    
    
    #Show the confusion matrix
    matrix = confusion_matrix(test_label, test_result)