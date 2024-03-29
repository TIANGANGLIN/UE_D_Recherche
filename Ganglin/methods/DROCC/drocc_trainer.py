import os
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

#trainer class for DROCC
class DROCCTrainer:
    """
    Trainer class that implements the DROCC algorithm proposed in
    https://arxiv.org/abs/2002.12718
    """

    def __init__(self, model, optimizer, lamda, radius, gamma, device):
        """Initialize the DROCC Trainer class

        Parameters
        ----------
        model: Torch neural network object
        optimizer: Total number of epochs for training.
        lamda: Weight given to the adversarial loss
        radius: Radius of hypersphere to sample points from.
        gamma: Parameter to vary projection.
        device: torch.device object for device to use.
        """     
        self.model = model
        self.optimizer = optimizer
        self.lamda = lamda
        self.radius = radius
        self.gamma = gamma
        self.device = device

        # Initialization pf tensorboard
        self.log_dir = "log"
        self.writer = SummaryWriter(self.log_dir)

    def train(self, train_loader, val_loader, learning_rate, lr_scheduler, total_epochs, 
                only_ce_epochs=50, ascent_step_size=0.001, ascent_num_steps=50,
                metric='AUC'):
        """Trains the model on the given training dataset with periodic 
        evaluation on the validation dataset.

        Parameters
        ----------
        train_loader: Dataloader object for the training dataset.
        val_loader: Dataloader object for the validation dataset.
        learning_rate: Initial learning rate for training.
        total_epochs: Total number of epochs for training.
        only_ce_epochs: Number of epochs for initial pretraining.
        ascent_step_size: Step size for gradient ascent for adversarial 
                          generation of negative points.
        ascent_num_steps: Number of gradient ascent steps for adversarial 
                          generation of negative points.
        metric: Metric used for evaluation (AUC / F1).
        """
        best_score = -np.inf
        best_model = None
        self.ascent_num_steps = ascent_num_steps
        self.ascent_step_size = ascent_step_size
        for epoch in range(total_epochs): 
            #Make the weights trainable
            self.model.train()
            lr_scheduler(epoch, total_epochs, only_ce_epochs, learning_rate, self.optimizer)
            
            #Placeholder for the respective 2 loss values
            epoch_adv_loss = torch.tensor([0]).type(torch.float32).to(self.device)  #AdvLoss
            epoch_ce_loss = 0  #Cross entropy Loss
            
            batch_idx = -1
            for data, target, _ in train_loader:
                batch_idx += 1
                data, target = data.to(self.device), target.to(self.device)
                # Data Processing
                data = data.to(torch.float)
                target = target.to(torch.float)
                target = torch.squeeze(target)
                
                # b = data[0].permute(1,2,0).cpu().numpy()
                # plt.imshow(b)
                # plt.savefig("test.png")
                self.optimizer.zero_grad()
                
                # Extract the logits for cross entropy loss
                logits = self.model(data)
                logits = torch.squeeze(logits, dim = 1)
                ce_loss = F.binary_cross_entropy_with_logits(logits, target)
                # Add to the epoch variable for printing average CE Loss
                epoch_ce_loss += ce_loss

                '''
                Adversarial Loss is calculated only for the positive data points (label==1).
                '''
                if  epoch >= only_ce_epochs:
                    data = data[target == 1]
                    # AdvLoss 
                    adv_loss = self.one_class_adv_loss(data)
                    epoch_adv_loss += adv_loss

                    loss = ce_loss + adv_loss * self.lamda
                else: 
                    # If only CE based training has to be done
                    loss = ce_loss
                
                # Backprop
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('Train/Loss', loss, epoch)
                self.writer.flush()
                    
            epoch_ce_loss = epoch_ce_loss/(batch_idx + 1)  #Average CE Loss
            epoch_adv_loss = epoch_adv_loss/(batch_idx + 1) #Average AdvLoss

            test_score = self.test(val_loader, metric, "cifra10")
            if test_score > best_score:
                best_score = test_score
                best_model = copy.deepcopy(self.model)
            print('Epoch: {}, CE Loss: {}, AdvLoss: {}, {}: {}'.format(
                epoch, epoch_ce_loss.item(), epoch_adv_loss.item(), 
                metric, test_score))
        self.model = copy.deepcopy(best_model)
        print('\nBest test {}: {}'.format(
            metric, best_score
        ))

    def test(self, test_loader, metric, dataset_name):
        """Evaluate the model on the given test dataset.

        Parameters
        ----------
        test_loader: Dataloader object for the test dataset.
        metric: Metric used for evaluation (AUC / F1).
        """        
        self.model.eval()
        label_score = []
        batch_idx = -1

        # Loade Kaggle dataset 
        from torchvision import transforms
        import torchvision
        normalize_k = transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
        transform_k  = transforms.Compose([
                #  transforms.RandomResizedCrop(224),
                transforms.Resize([32,32]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_k,])

        idx_label_score = []
        if dataset_name=="cifra10":
            for data, target, idx in test_loader:
                batch_idx += 1
                data, target = data.to(self.device), target.to(self.device)
                data = data.to(torch.float)
                target = target.to(torch.float)
                target = torch.squeeze(target)

                logits = self.model(data)
                logits = torch.squeeze(logits, dim = 1)
                sigmoid_logits = torch.sigmoid(logits)
                scores = logits
                label_score += list(zip(target.cpu().data.numpy().tolist(), scores.cpu().data.numpy().tolist()))
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                                target.cpu().data.numpy().tolist(),
                                                scores.cpu().data.numpy().tolist()))
        else:
            if dataset_name=="cartoon_dog":
                dataset_k = torchvision.datasets.ImageFolder(root='./Ganglin/data/cartoon_dog',transform=transform_k)
            elif dataset_name=="kaggle_testset":
                dataset_k = torchvision.datasets.ImageFolder(root='./Ganglin/data/kaggle_testset',transform=transform_k)
            
            kaggle_test_loader = torch.utils.data.DataLoader(
                                dataset_k,
                                batch_size=test_loader.batch_size,
                                shuffle=True)

            os.makedirs("tmp",exist_ok=1)
            for kaggle in kaggle_test_loader:
                if dataset_k.class_to_idx['dogs']==1:
                    data, target = kaggle[0], kaggle[1]
                else:
                    data, target = kaggle[0], abs(kaggle[1]-1)
                batch_idx += 1
                data, target = data.to(self.device), target.to(self.device)
                #########################################
                # for l in range(len(data)):
                #     img = data[l].permute(1,2,0).cpu().numpy()
                #     plt.imshow(img)
                #     plt.savefig("tmp/{}.png".format(str(l).zfill(3)))
                #########################################
                data = data.to(torch.float)
                target = target.to(torch.float)
                target = torch.squeeze(target)

                logits = self.model(data)
                logits = torch.squeeze(logits, dim = 1)
                sigmoid_logits = torch.sigmoid(logits)
                scores = logits
                label_score += list(zip(target.cpu().data.numpy().tolist(), scores.cpu().data.numpy().tolist()))


        try:
            idx, labels, scores = zip(*idx_label_score)
            idx = np.array(idx)
            labels = np.array(labels)
            scores = np.array(scores)
            # save for comparaison
            save_labels_scores = {}
            save_labels_scores['idx'] = idx
            save_labels_scores['labels'] = labels
            save_labels_scores['scores'] = scores
            os.makedirs("results",exist_ok=1)
            np.save("results/save_labels_scores.npy",save_labels_scores)
        
        except:
            # Compute test score
            labels, scores = zip(*label_score)
            labels = np.array(labels)
            scores = np.array(scores)
        
        

        if metric == 'F1':
            # Evaluation based on https://openreview.net/forum?id=BJJLHbb0-
            thresh = np.percentile(scores, 20)
            y_pred = np.where(scores >= thresh, 1, 0)
            prec, recall, test_metric, _ = precision_recall_fscore_support(
                labels, y_pred, average="binary")
        if metric == 'AUC':
            test_metric = roc_auc_score(labels, scores)
        return test_metric
        
    def test_one_img(self, img, metric):
        """Evaluate the model on the given test dataset.

        Parameters
        ----------
        test_loader: Dataloader object for the test dataset.
        metric: Metric used for evaluation (AUC / F1).
        """        
        self.model.eval()
        label_score = []
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.247, 0.243, 0.261])])

        img = transform(img)
        img = img.unsqueeze(0)
        data = img.to(self.device)
        data = data.to(torch.float)
        target = torch.ones([1]).to(torch.float)
        target = torch.squeeze(target)

        logits = self.model(data)
        logits = torch.squeeze(logits, dim = 1)
        sigmoid_logits = torch.sigmoid(logits)
        scores = logits
        label_score += list(zip([target.cpu().data.numpy().tolist()],
                                        scores.cpu().data.numpy().tolist()))
        # Compute test score
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        if metric == 'F1':
            # Evaluation based on https://openreview.net/forum?id=BJJLHbb0-
            thresh = np.percentile(scores, 20)
            y_pred = np.where(scores >= thresh, 1, 0)
            prec, recall, test_metric, _ = precision_recall_fscore_support(
                labels, y_pred, average="binary")
        if metric == 'AUC':
            test_metric = roc_auc_score(labels, scores)
        return test_metric
        
    
    def one_class_adv_loss(self, x_train_data):
        """Computes the adversarial loss:
        1) Sample points initially at random around the positive training
            data points
        2) Gradient ascent to find the most optimal point in set N_i(r) 
            classified as +ve (label=0). This is done by maximizing 
            the CE loss wrt label 0
        3) Project the points between spheres of radius R and gamma * R 
            (set N_i(r))
        4) Pass the calculated adversarial points through the model, 
            and calculate the CE loss wrt target class 0
        
        Parameters
        ----------
        x_train_data: Batch of data to compute loss on.
        """
        batch_size = len(x_train_data)
        # Randomly sample points around the training data
        # We will perform SGD on these to find the adversarial points
        x_adv = torch.randn(x_train_data.shape).to(self.device).detach().requires_grad_()
        x_adv_sampled = x_adv + x_train_data

        for step in range(self.ascent_num_steps):
            with torch.enable_grad():

                new_targets = torch.zeros(batch_size, 1).to(self.device)
                new_targets = torch.squeeze(new_targets)
                new_targets = new_targets.to(torch.float)
                
                logits = self.model(x_adv_sampled)         
                logits = torch.squeeze(logits, dim = 1)
                new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)

                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                grad_norm = torch.norm(grad, p=2, dim = tuple(range(1, grad.dim())))
                grad_norm = grad_norm.view(-1, *[1]*(grad.dim()-1))
                grad_normalized = grad/grad_norm 
            with torch.no_grad():
                x_adv_sampled.add_(self.ascent_step_size * grad_normalized)

            if (step + 1) % 10==0:
                # Project the normal points to the set N_i(r)
                h = x_adv_sampled - x_train_data
                norm_h = torch.sqrt(torch.sum(h**2, 
                                                dim=tuple(range(1, h.dim()))))
                alpha = torch.clamp(norm_h, self.radius, 
                                    self.gamma * self.radius).to(self.device)
                # Make use of broadcast to project h
                proj = (alpha/norm_h).view(-1, *[1] * (h.dim()-1))
                h = proj * h
                x_adv_sampled = x_train_data + h  #These adv_points are now on the surface of hyper-sphere

        adv_pred = self.model(x_adv_sampled)
        adv_pred = torch.squeeze(adv_pred, dim=1)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets * 0))

        return adv_loss

    def save(self, path):
        print("Save model in: ",os.path.join(path, 'model.pt'))
        torch.save(self.model.state_dict(),os.path.join(path, 'model.pt'))

    def load(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
