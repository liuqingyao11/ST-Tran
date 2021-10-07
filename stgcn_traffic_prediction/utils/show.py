import numpy as np
import matplotlib.pyplot as plt
import torch
import math

# Fixing random state for reproducibility
np.random.seed(19680801)
 
x1 = torch.rand((40,3))
X_in = [x1]*3
x2 = torch.rand((40,3))
X_out = [x2]*3
x3 = torch.rand((40,3))
T = [x3]*3

def compare(pred_fs,pred_wofs,tgt,filename=None):
    # len*c
    fig,axs = plt.subplots(1,1)
    during = len(pred_fs)
    t=np.arange(0,during,1)
    s1 = pred_wofs[:,0]
    s2 = pred_fs[:,0]
    s3 = tgt[:,0]

    axs.plot(t, s1, 'b',label='with STB')
    axs.plot(t, s2, 'g',label='without STB')
    axs.plot(t, s3, 'r--',label='ground truth')
    axs.set_xlabel('time')
    axs.set_ylabel('traffic data')
    plt.legend(loc='upper right')
    axs.grid(True)
    plt.savefig(filename+'.png')
    plt.show()

def plot(pred,tgt,filename=None):
    # len*c
    fig,axs = plt.subplots(1,1)
    during = len(pred)
    t=np.arange(0,during,1)
    #s1 = pred_wofs[:,0]
    s2 = pred[:,0]
    s3 = tgt[:,0]
    axs.plot(t, s2, 'g',label='prediction')
    axs.plot(t, s3, 'r--',label='ground truth')
    axs.set_xlabel('time')
    axs.set_ylabel('traffic data')
    plt.legend(loc='upper right')
    axs.grid(True)
    plt.savefig(filename+'.png')
    plt.show()