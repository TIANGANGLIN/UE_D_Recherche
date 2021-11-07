import os 
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import sys 
sys.path.insert(0,"./Ganglin/methods/Deep-SVDD-PyTorch/src")
from datasets.main import load_dataset
from utils.visualization.plot_images_grid import plot_images_grid

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)
dataset_name = "cifar10"
data_path = "./Ganglin/data"
xp_path = "results_compare"
os.makedirs(xp_path,exist_ok=1)

deepSVDD_path = "./Ganglin/methods/Deep-SVDD-PyTorch/src/results/save_labels_scores.npy"
drocc_path = "./Ganglin/methods/DROCC/results/save_labels_scores.npy"
ganomaly_path = "./Ganglin/methods/ganomaly/results/save_labels_scores.npy"

deepSVDD = np.load(deepSVDD_path, allow_pickle=1).item()
drocc = np.load(drocc_path, allow_pickle=1).item()
ganomaly = np.load(ganomaly_path, allow_pickle=1).item()


dataset = load_dataset(dataset_name, data_path, 5)

def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma


normal_scores = []
def cal_aur(method,method_name):
    if not isinstance(method['idx'],torch.Tensor):
        indices, labels, scores = np.array(method['idx']), np.array(method['labels']), np.array(method['scores'])
    else:
        indices, labels, scores = np.array(ganomaly['idx'].cpu().numpy()), np.array(ganomaly['labels'].cpu().numpy()), np.array(ganomaly['scores'].cpu().numpy())

    scores = (scores - min(scores)) / (max(scores) - min(scores))
    if method_name == "deepSVDD" or method_name == "ganomaly":
        scores = standardization(scores)
    else:
        scores = standardization(1-scores)

    normal_scores.append(scores[labels == 1])
    idx_sorted = indices[labels == 1][np.argsort(scores[labels == 1])]  # sorted from lowest to highest anomaly score
    X_normals = torch.tensor(np.transpose(dataset.test_set.data[idx_sorted[:32], ...], (0, 3, 1, 2)))
    X_outliers = torch.tensor(np.transpose(dataset.test_set.data[idx_sorted[-32:], ...], (0, 3, 1, 2)))

    plot_images_grid(X_normals, export_img=xp_path + '/normals_{}'.format(method_name), title='Most normal examples', padding=2)
    plot_images_grid(X_outliers, export_img=xp_path + '/outliers_{}'.format(method_name), title='Most anomalous examples', padding=2)
    
    # if method_name == "deepSVDD" or method_name == "ganomaly":
    print("AUR: {:.4f}".format(roc_auc_score(1-labels, scores)))
    # else:
        # print("AUR: {:.5f}".format(roc_auc_score(labels, scores)))

    return scores

deepSVDD_scores = cal_aur(deepSVDD,"deepSVDD")
drocc_scores = cal_aur(drocc,"drocc")
ganomaly_scores = cal_aur(ganomaly,"ganomaly")

print(np.corrcoef(deepSVDD_scores, drocc_scores))
print(np.corrcoef(deepSVDD_scores, ganomaly_scores))
print(np.corrcoef(drocc_scores, ganomaly_scores))

scores_12 = np.vstack([deepSVDD_scores,drocc_scores])
scores_123 = np.vstack([scores_12,ganomaly_scores])
rho = np.corrcoef(scores_123)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(24, 6))

ax[0].scatter(scores_123[0,],scores_123[1,])
ax[0].title.set_text('deepSVDD and drocc Correlation = ' + "{:.2f}".format(rho[0,1]))
ax[0].set(xlabel='x',ylabel='y')

ax[1].scatter(scores_123[0,],scores_123[2,])
ax[1].title.set_text('deepSVDD and ganomaly Correlation = ' + "{:.2f}".format(rho[0,2]))
ax[1].set(xlabel='x',ylabel='y')

ax[2].scatter(scores_123[1,],scores_123[2,])
ax[2].title.set_text('drocc and ganomaly Correlation = ' + "{:.2f}".format(rho[1,2]))
ax[2].set(xlabel='x',ylabel='y')


fig.subplots_adjust(wspace=.4)    
plt.show()
plt.savefig("correlation.png")

print(roc_auc_score(1-deepSVDD['labels'], (deepSVDD_scores+drocc_scores)))
print(roc_auc_score(1-deepSVDD['labels'], (deepSVDD_scores+ganomaly_scores)))
print(roc_auc_score(1-deepSVDD['labels'], (drocc_scores+ganomaly_scores)))
print(roc_auc_score(1-deepSVDD['labels'], (deepSVDD_scores+drocc_scores+ganomaly_scores)))





de = 0


normal_scores