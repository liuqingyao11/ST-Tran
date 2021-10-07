import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs


def get_adj(nums):
    A = np.zeros((int(nums), int(nums)), dtype = np.float32)
    stride = int(np.sqrt(nums))
    for i in range(nums):
        A[i][i]=1.0
        if not i-1<0:
            A[i][i-1]=1.0
        if not i+1>nums-1:
            A[i][i+1] = 1.0
        if not i-stride<0:
            A[i][i-stride] = 1.0
        if not i+stride>nums-1:
            A[i][i+stride] = 1.0
    #print(A)
    return A

def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    
    '''
    
    assert W.shape[0] == W.shape[1]
    
    D = np.diag(np.sum(W, axis = 1))
    
    L = D - W
    
    lambda_max = eigs(L, k = 1, which = 'LR')[0].real
    
    return (2 * L) / lambda_max - np.identity(W.shape[0])

def getxy(x,m):
    index = torch.zeros(2,dtype=torch.float32)
    index[0] = x//m
    index[1] = x%m
    return index

def getD(m):
    D = torch.ones((m,m),dtype=torch.float32)*1.5
    for i in range(m):
        for j in range(i):
            D[i,j] = 1/torch.norm((getxy(i,m)-getxy(j,m)),2)
            D[j,i] = D[i,j]
    return D


def getA_cosin(x):
    (bs,flow,N,c) = x.shape
    x = x.transpose(1,2).contiguous().view((bs,N,c*2))

    normed = torch.norm(x,2,dim=-1).unsqueeze(-1)
    print(normed.shape)
    tnormed = normed.transpose(1,2)
    A = x.matmul(x.transpose(1,2))/normed.matmul(tnormed)
    return F.softmax(A,dim=-1)


def getA_corr(x):
    (bs,flow,N,c) = x.shape
    x = x.transpose(1,2).contiguous().view((bs,N,c*2))
    A = torch.zeros((bs,N,N),dtype=torch.float32,requires_grad=False)
    for i in range(bs):
        A[i] = torch.from_numpy(np.absolute(np.corrcoef(x[i].numpy())))
    for j in range(N): 
        A[:,j,j] = -1e9

    return F.softmax(A.reshape(bs,1,-1),dim=-1).reshape(bs,N,N)

def getadj(x):
    (bs,flow,N,c) = x.shape
    x = x.transpose(1,2).contiguous().view((bs,N,c*2)).numpy()
    A = np.zeros((bs,N,N),dtype=np.float32)
    for i in range(bs):
        A[i] = np.absolute(np.corrcoef(x[i]))
        A[i] = scaled_Laplacian(A[i])
        D = np.array(np.sum(A[i],axis=-1))
        D = np.matrix(np.diag(D))
        A[i] = D**-1*A[i]
        A[i][np.isnan(A[i])] = 0.
    print(A)
    return torch.from_numpy(A).cuda()


def c_subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def p_subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.zeros(np.ones(attn_shape)).astype('uint8')
    subsequent_mask[:,:,0] = 1
    return torch.from_numpy(subsequent_mask) == 0
