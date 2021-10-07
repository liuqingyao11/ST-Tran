import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
 
from stgcn_traffic_prediction.pygcn.layers import GraphConvolution
from .period import period
from .closeness import close
from .spatial import Spatial,gcnSpatial
from .utils import getadj,getA_cosin,getA_corr

class Fusion(nn.Module):
    def __init__(self,dim_in):
        super(Fusion,self).__init__()
        self.weight2 = nn.Linear(dim_in*2,dim_in)

    def forward(self,x1,x2=None):
        if(x2 is not None):
            out = self.weight2(torch.cat([x1,x2],dim=-1))
        else:
            out = x1
        return out

class T_STGCN(nn.Module):
    def __init__(self,len_closeness, external_size, N, k, spatial, s_model_d,c_model_d,p_model_d,t_model_d,dim_hid=16, drop_rate=0.1):
        super(T_STGCN,self).__init__()
        if(spatial=='gcn'):
            self.spatial = gcnSpatial(len_closeness,dim_hid,len_closeness,dropout=0.1)
        else:
            self.spatial = Spatial(len_closeness,k,N,s_model_d)
        self.c_temporal = close(k,N,c_model_d)
        self.p_temporal = period(len_closeness,N,p_model_d)
        #self.t_temporal = period(len_closeness,N,t_model_d)

        self.temporal_fusion = Fusion(len_closeness)
        if(spatial=='gcn'):
            self.spatial_f = gcnSpatial(len_closeness,dim_hid,len_closeness,dropout=0.1)
        else:
            self.spatial_f = Spatial(len_closeness,k,N,s_model_d)
        self.fusion = Fusion(len_closeness)
        self.k = k


    def forward(self,x_c,mode,c,s,FS,c_tgt,s_tgt,flow,x_p,x_t=None):
        '''initial data size
        x_c: bs*closeness*2*N
        x_p: bs*len_period*closeness*2*N
        x_t: bs*len_trend*closeness*2*N
        '''
        '''spatial output
        sx_c: bs*N*closeness
        '''
        '''temporal output
        sq_c: bs*N*closeness
        sq_p: bs*N*closeness
        sq_t: bs*N*closeness
        '''
        '''fused output
        bs*closeness*N
        ''' 

        bs = len(x_c)
        N = x_c.shape[-1]
        len_closeness = x_c.shape[1]
        x_spatial = None
        sq_t,sq_p,sq_c = None,None,None
        #print('x_c\n',x_c)

        #get adj
        if(mode=='cos'):
            adj = getA_cosin(x_c.permute((0,2,3,1)))
        elif(mode=='corr'):
            adj = getA_corr(x_c.permute((0,2,3,1)))
        else:
            raise Exception('wrong adj mode')
        index = torch.argsort(adj,dim=-1,descending=True)[:,:,0:self.k]
        if(s):
            #spatial
            x_spatial,_ = self.spatial(x_c,x_p,s_tgt,mode,flow,adj,index,x_t)
            #print('spatial:',x_spatial[0])

        #temporal
        if(c):
            sq_c = F.sigmoid(self.c_temporal(x_c,x_p,c_tgt,mode,flow,adj,index,x_t))
            #print('sq_c:',sq_c[0])

        sq_p = self.p_temporal(x_c, x_p,flow)
        #print('period:',sq_p.shape)

        # if x_t is not None:
        #     sq_t = self.t_temporal(x_c, x_t,flow)
        #     #print('trend:',sq_t[0])
        x_temporal = self.temporal_fusion(sq_p,sq_c)
        
        if(FS):
            x_temporal,_ = self.spatial_f(x_c,x_temporal.transpose(1,2).unsqueeze(-2).unsqueeze(1),'p',mode,flow,adj,index,x_t)

        #fusion
        pred = self.fusion(x_temporal,x_spatial)
        return pred.transpose(1,2)



