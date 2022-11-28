from biodatasets import load_dataset
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
from sampler import Sampler
import os
from network import ConvARG
from torch.utils.data import DataLoader
import torch.nn as nn
from time import time
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

# model
net = ConvARG(23).cuda()
opt = torch.optim.Adam(net.parameters(), lr=leaing_rate)

# train val dataset
portion = int(np.floor(len(X)*0.9))
X_train, Y_train = X_oh[0:portion], y[0:portion]
X_val, Y_val = X_oh[portion:-1], y[portion:-1]
train_ds, val_ds = Sampler(X_train, Y_train), Sampler(X_val, Y_val)
train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
# loss function
loss_func = nn.CrossEntropyLoss()
# learning rate decay, >=3rd*0.1, >=100th*0.01
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [3, 100])

# train network
start = time()
for epoch in range(Epoch):
    lr_decay.step()
    mean_loss = []
    net.train(True)
    for step, (X, y) in enumerate(train_dataloader):
        X = X.cuda()
        y = y.cuda()

        output = net(X)
        # print(y.shape,output[:,0].shape)
        loss = loss_func(output, y)
        # break
        mean_loss.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 4 == 0:
            print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss.item(), (time() - start) / 60))

    mean_loss = sum(mean_loss) / len(mean_loss)

    # save model every 10 epoches
    # naming of DN：#epoch+ validation mean loss + train mean loss + validation accuracy
    if epoch % 10 == 0:
        running_vloss = 0.0
        running_vacc = 0.0
        net.train(False)
        for i, vdata in enumerate(val_dataloader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.cuda(), vlabels.cuda()
            voutputs = net(vinputs)
            vloss = loss_func(voutputs, vlabels)
            running_vloss += vloss
            # acc
            pred = voutputs[:,1]
            # print(pred.shape)
            pred[pred>=0.5] = 1
            pred[pred<0.5] = 0
            val_acc = torch.sum(pred == vlabels)
            running_vacc += val_acc
        avg_vloss = running_vloss / (i + 1)
        avg_vacc = running_vacc / ((i+1)*batch_size)
        print('LOSS train {} valid {}'.format(mean_loss, avg_vloss))
        torch.save(net.state_dict(), './repository/net{}-vloss{:.3f}-tloss{:.3f}-vacc{:.3f}-24448.pth'.format(epoch, avg_vloss, mean_loss, avg_vacc))




