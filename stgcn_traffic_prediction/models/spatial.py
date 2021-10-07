import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from stgcn_traffic_prediction.pygcn.models import GCN
from stgcn_traffic_prediction.models.transformer import make_model
from .utils import getA_cosin,getA_corr,getadj,get_adj,scaled_Laplacian

class gcnSpatial(nn.Module):
    def __init__(self,dim_in,dim_hid,dim_out,dropout):
        super(gcnSpatial,self).__init__()
        self.spatial = GCN(dim_in,dim_hid,dim_out,dropout)

    def forward(self,x_c,x_p,tgt_mode,mode,flow,A=None,index=None,x_t=None):
        #print('x_c:',x_c)
        N = x_c.shape[-1]
        sx_c = x_c.permute(0,2,3,1).float()
        #print('sx',sx_c.shape)
        adj_mx = get_adj(N)
        L_tilde = torch.tensor(scaled_Laplacian(adj_mx)).float()
        #adj = getadj(sx_c)
        #print('gcn_adj',adj.shape)
        spatial_c = self.spatial(sx_c[:,flow].cuda(),L_tilde.cuda())
        return  spatial_c,adj_mx
 
class Spatial(nn.Module):
    def __init__(self,close_size,k,N,model_d):
        super(Spatial,self).__init__()
        self.spatial = make_model(close_size,close_size,N,model_d,spatial=True)
        self.k = k
 
    def forward(self,x_c,x_p,tgt_mode,mode,flow,A=None,index=None,x_t=None):
        '''initial data size
        x_c: bs*closeness*2*N
        x:   bs*2*N*closeness
        '''
        '''spatial input
        sx_c: bs*2*N*k*closeness
        tgt: bs*2*N*1*closeness
        '''
        '''spatial output
        sq_c: bs*N*1*closeness
        '''
        bs,closeness,_,N = x_c.shape
        x = x_c.permute((0,2,3,1)).float()
        #print('x',x.shape)
        #calculate the similarity between other nodes
        if A is None:
            if(mode=='cos'):
                A = getA_cosin(x)
            elif(mode=='moran'):
                A = getA_Moran(x)
            elif(mode=='corr'):#maybe need absolute
                A = getA_corr(x)
            else:
                raise Exception('wrong adj mode')
            #A.shape=bs,N,N
        #print('A',A.shape)
        #selected top-k node

        sx_c = torch.zeros((bs,N,self.k,closeness),dtype=torch.float32)
        if index is None:
            index = torch.argsort(A,dim=-1,descending=True)[:,:,0:self.k] #bs,N,k
        # selected_c = []
        for i in range(bs):
            for j in range(N):
                sx_c[i,j] = x[i,flow,index[i,j]]
        #sx_c = torch.cat(selected_c,dim=2).cuda()
        #print('sx_c',sx_c.shape)
        #sx_c:(bs,N,k,closeness)

        if(tgt_mode=='c'):
            tgt = x[:,flow].unsqueeze(dim=-2).cuda()
            #tgt_c = sx_c[:,flow].unsqueeze(-1).cuda()
        elif(tgt_mode=='r'):
            tgt = torch.rand((bs,N,1,closeness)).cuda()
        elif(tgt_mode=='p'):
            #print('before tgt_p',x_p.shape)
            tgt = torch.mean(x_p[:,:,:,flow],dim=1).transpose(1,2).unsqueeze(-2).cuda()
        elif(tgt_mode=='t'):
            tgt = torch.mean(x_t[:,:,:,flow],dim=1).transpose(1,2).unsqueeze(-2).cuda()
        #spatial transformer
        
       # print('s_tgt',tgt.shape)
        sq_c = self.spatial(sx_c.cuda(), tgt).squeeze(dim=-2)
        #print('sq_c',sq_c.shape)
        #return sq_c.permute((0,3,1,2)) 
        #return F.sigmoid(sq_c).permute((0,3,1,2))
        return sq_c,tgt.permute((0,2,3,1)).squeeze(1)
 