from biodatasets import load_dataset
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
from sampler import Sampler
import os
from network import ConvARG
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from utils import *

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# load dataset
dataset = load_dataset("antibiotic-resistance")

# Get data and shuffle
X, y = dataset.to_npy_arrays(input_names=["sequence"], target_names=["label"])
X,y=X[0],y[0]
X, y = unison_shuffled_copies(X,y)

# 1. one hot encoding
X_oh = encode_one_hot(X)
X_oh.shape
# 2. cotinious embedding
# cls_embeddings = dataset.get_embeddings("sequence", "esm1-t34-670M-UR100", "cls")
# cls_embeddings

# Hyperparamter
on_server = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0' if on_server is False else '1,2,3'
cudnn.benchmark = True
Epoch = 300
leaing_rate = 1e-4
batch_size = 64
module_dir = './repository/net170-vloss0.344-tloss0.327-vacc0.968-49664.pth'

# model
net = ConvARG(23).cuda()
net.load_state_dict(torch.load(module_dir))
net.train(False)

# train val dataset
portion = int(np.floor(len(X)*0.9))
X_train, Y_train = X_oh[0:portion], y[0:portion]
X_val, Y_val = X_oh[portion:-1], y[portion:-1]
train_ds, val_ds = Sampler(X_train, Y_train), Sampler(X_val, Y_val)
val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

# validation
running_vloss = 0.0
running_vacc = 0.0
for i, vdata in enumerate(val_dataloader):
    # 1. make prediction on validation data
    vinputs, vlabels = vdata
    vinputs, vlabels = vinputs.cuda(), vlabels.cuda()
    voutputs = net(vinputs)
    # 2. calculate validation accuracy 
    pred = voutputs[:,1]
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    val_acc = torch.sum(pred == vlabels)
    running_vacc += val_acc
avg_vacc = running_vacc / ((i+1)*batch_size)
print('Validation accuracy {}'.format(avg_vacc))