from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys 
sys.path.insert(0,"./Ganglin/methods/Deep-SVDD-PyTorch/src")
from datasets.preprocessing import get_target_label_idx, global_contrast_normalization

class Novelty_MNIST():
    def __init__(self, normal_class=0, downloads=True, train_amount=60000, test_amount=2000):
        self.downloads    = downloads
        self.train_amount = train_amount
        self.test_amount  = test_amount
        self.normal_class = normal_class
        # Pre-computed min and max values (after applying GCN) from train data per class
        self.min_max = [(-0.8826567065619495, 9.001545489292527),
                        (-0.6661464580883915, 20.108062262467364),
                        (-0.7820454743183202, 11.665100841080346),
                        (-0.7645772083211267, 12.895051191467457),
                        (-0.7253923114302238, 12.683235701611533),
                        (-0.7698501867861425, 13.103278415430502),
                        (-0.778418217980696, 10.457837397569108),
                        (-0.7129780970522351, 12.057777597673047),
                        (-0.8280402650205075, 10.581538445782988),
                        (-0.7369959242164307, 10.697039838804978)]

        self.normal_classes = tuple([self.normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(self.normal_class)

        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        self.transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([self.min_max[self.normal_class][0]],
                                                            [self.min_max[self.normal_class][1] - self.min_max[self.normal_class][0]])])

        self.target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

    def MNIST_DATASET_TRAIN(self):
        """
        Inputs:

        Outputs:

        """
        # Load dataset
        train_set = torchvision.datasets.MNIST(root = 'data/mnist/', 
                    train = True, 
                    download = self.downloads, 
                    transform=self.transform, 
                    target_transform=self.target_transform
                    )
                                
        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.targets.clone().data.cpu().numpy(), self.normal_classes)
        training_data = Subset(train_set, train_idx_normal)

        #Convert Training data to numpy
        train_data = train_set.data[train_idx_normal,:,:].numpy()[:self.train_amount]
        train_label = train_set.targets[train_idx_normal].numpy()[:self.train_amount]
        train_label_final = np.ones_like(train_label)
        train_label_final[train_label!=self.normal_class] = -1
        # Print training data size
        # print('Training data size: ',train_data.shape)
        # print('Training data label size:',train_label.shape)   
        # plt.imshow(train_data[0])
        # plt.show()
        
        train_data = train_data/255.0
        
        return train_data, train_label_final


    def MNIST_DATASET_TEST(self):
        # Load dataset
        testing_data = torchvision.datasets.MNIST(
                root = 'data/mnist/',
                train = False, # transform = torchvision.transforms.ToTensor(),
                transform=self.transform, target_transform=self.target_transform,
                download = self.downloads
                )
        
        # Convert Testing data to numpy
        test_data = testing_data.data.numpy()[:self.test_amount]
        test_label = testing_data.targets.numpy()[:self.test_amount]
        test_label_final = np.ones_like(test_label)
        test_label_final[test_label!=self.normal_class] = -1
        
        # Print training data size
        # print('test data size: ',test_data.shape)
        # print('test data label size:',test_label.shape)   
        # plt.imshow(test_data[0])
        # plt.show()
        
        test_data = test_data/255.0
        
        return test_data, test_label_final


if __name__=='__main__':
    """
    Debug Novelty_MNIST()
    """
    mnist = Novelty_MNIST()

    train_data, train_label = mnist.MNIST_DATASET_TRAIN()
    test_data, test_label = mnist.MNIST_DATASET_TEST()
    print('test data size: ',test_data.shape)
    print('test data label size:',test_label.shape)   
    plt.imshow(test_data[0])
    plt.show()