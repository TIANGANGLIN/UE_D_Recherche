from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
from datasets.preprocessing import get_target_label_idx, global_contrast_normalization
import matplotlib.pyplot as plt
import numpy as np


class Novelty_CIFAR10():
    def __init__(self, normal_class=0, downloads=True):
        self.downloads    = downloads
        self.normal_class = normal_class
        
        self.normal_classes = tuple([self.normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(self.normal_class)

        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        self.transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.2, 0.2, 0.2])])

        self.target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

    def DATASET_TRAIN(self):
        # Load dataset
        train_set = torchvision.datasets.CIFAR10(root = 'data/cifra10/', 
                    train = True, 
                    download = self.downloads, 
                    transform=self.transform, 
                    target_transform=self.target_transform
                    )
                                
        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.targets, self.normal_classes)
        training_data = Subset(train_set, train_idx_normal)

        #Convert Training data to numpy
        train_data = train_set.data[train_idx_normal,:,:,:]
        train_label = np.array(train_set.targets)[train_idx_normal]
        train_label_final = np.ones_like(train_label)
        train_label_final[train_label!=self.normal_class] = -1
        # Print training data size
        # print('Training data size: ',train_data.shape)
        # print('Training data label size:',train_label.shape)   
        # plt.imshow(train_data[0])
        # plt.show()
        
        # train_data = train_data/255.0
        
        return train_data, train_label_final


    def DATASET_TEST(self):
        # Load dataset
        testing_data = torchvision.datasets.CIFAR10(
                root = 'data/cifra10/',
                train = False, # transform = torchvision.transforms.ToTensor(),
                transform=self.transform, target_transform=self.target_transform,
                download = self.downloads
                )
        
        # Convert Testing data to numpy
        test_data = testing_data.data
        test_label = np.array(testing_data.targets)
        test_label_final = np.ones_like(test_label)
        test_label_final[test_label!=self.normal_class] = -1
        
        # Print training data size
        # print('test data size: ',test_data.shape)
        # print('test data label size:',test_label.shape)   
        # plt.imshow(test_data[0])
        # plt.show()
        
        
        return test_data, test_label_final


if __name__=='__main__':
    """
    Debug Novelty_Novelty_CIFAR10()
    """
    cifra10 = Novelty_CIFAR10()

    train_data, train_label = cifra10.DATASET_TRAIN()
    test_data, test_label = cifra10.DATASET_TEST()
    print('test data size: ',test_data.shape)
    print('test data label size:',test_label.shape)   
    plt.imshow(test_data[0])
    plt.show()