import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from stgcn_traffic_prediction.models.transformer import make_model
from stgcn_traffic_prediction.models.utils import c_subsequent_mask,getA_cosin,getA_corr

class close(nn.Module):
    def __init__(self,k,N,model_d):
        super(close,self).__init__()
        self.c_temporal = make_model(k+1,1,N,model_d)
        self.k = k

    def forward(self,x_c,x_p,tgt_mode,mode,flow,adj=None,index=None,x_t=None):
        '''initial data size
        x_c: bs*closeness*2*N
        sx_c:bs*2*N*closeness
        '''
        bs = len(x_c)
        N = x_c.shape[-1]
        len_closeness = x_c.shape[1]

        #adj
        sx_c = x_c.permute((0,2,3,1)).float()
        if adj is None:
            if(mode=='cos'):
                adj = getA_cosin(sx_c)
            elif(mode=='corr'):#maybe need absolute
                adj = getA_corr(sx_c)
            else:
                raise Exception('wrong adj mode')
        #print(adj.shape)
        if index is None: 
            index = torch.argsort(adj,dim=-1,descending=True)[:,:,0:self.k]
        selected = torch.zeros((bs,N,self.k,len_closeness),dtype=torch.float)
        for i in range(bs):
            for j in range(N):
                    selected[i,j] = sx_c[i,flow,index[i,j]]
        #(bs,N,k,c)

        tx_c = torch.cat([sx_c[:,flow].unsqueeze(-1),selected.transpose(-1,-2)],dim=-1).cuda()
        #(bs,N,c,k+1)


        '''temporal input
        tx_c: bs*N*closeness*(k+1)
        tgt:bs*N*closeness*1 
        '''
        '''temporal output
        sq_c: bs*N*closeness*1
        '''

        tgt_mask_c = c_subsequent_mask(len_closeness).cuda()
        if(tgt_mode=='c'):
            tgt_c = sx_c[:,flow].unsqueeze(-1).cuda()
        elif(tgt_mode=='r'):
            tgt_c = torch.rand((bs,N,len_closeness,1)).cuda()
        elif(tgt_mode=='p'):
            tgt_c = torch.mean(x_p[:,:,:,flow],dim=1).transpose(1,2).unsqueeze(-1).cuda()
        elif(tgt_mode=='t'):
            tgt_c = torch.mean(x_t[:,:,:,flow],dim=1).transpose(1,2).unsqueeze(-1).cuda()
        elif(tgt_mode=='tp'):
            tgt_c = torch.mean(x_p[:,:,:,flow]+x_t[:,:,:,flow],dim=1).transpose(1,2).unsqueeze(-1).cuda()

        sq_c = self.c_temporal(tx_c, tgt_c, tgt_mask_c).squeeze(-1)
        return sq_c

