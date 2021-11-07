import sys
sys.path.insert(0, "methods/kPCA")
from methods.kpca import kPCA
from methods.pca import PCA
from datasets.dataLoader import loader
from utils.utils import Seedy,param_heatmap,param_scatter
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from pyod.models.ocsvm import OCSVM


"""
PCA
"""
dataset_name = "MNIST"
x_train, x_val, y_val, x_test, y_test = loader(dataset_name)
num_params = x_train.shape[1]
qs = np.linspace(1,x_train.shape[1],num_params,dtype = 'int')

gridsearch = np.zeros((num_params,2))

#perform gridsearch
run = 0
for q in qs: 
    model = PCA(q)
    model.fit(x_train)
    val_scores = model.decision_function(x_val)
    auc = metrics.roc_auc_score(y_val,val_scores)
    gridsearch[run,:]= np.asarray([q,auc])
    run += 1

best_idx = np.argmax(gridsearch[:,1])      
val_auc = np.max(gridsearch[:,1]) 
best_q = gridsearch[best_idx,0]

if best_q == x_train.shape[1]:
    best_q -= 1 #don't produce only zeros

model = PCA(q = int(best_q))
model.fit(x_train)
test_scores = model.decision_function(x_test)
test_auc = metrics.roc_auc_score(y_test,test_scores)