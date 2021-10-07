import sklearn.metrics as metrics
import numpy as np
import torch

def getmetrics(pred,truth):
    print('get metrics with test data...')
    mae = metrics.mean_absolute_error(truth,pred)
    mse = metrics.mean_squared_error(truth,pred)
    rmse = mse**0.5
    nrmse = rmse/np.mean(truth)
    r2 = metrics.r2_score(truth,pred)
 
    return mae,mse,rmse,nrmse,r2 