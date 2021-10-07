import os
import sys
import argparse
import numpy as np
from datetime import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
sys.path.append('../../')
from stgcn_traffic_prediction.dataloader.milano_crop import load_data
from stgcn_traffic_prediction.models.model import T_STGCN
from stgcn_traffic_prediction.utils.lr_scheduler import LR_Scheduler
from stgcn_traffic_prediction.utils.parser import getparse
from stgcn_traffic_prediction.utils.metrics import getmetrics
from stgcn_traffic_prediction.utils.show import plot
import time

torch.manual_seed(22)
opt = getparse()
print(opt)
opt.save_dir = '{}/{}'.format(opt.save_dir, opt.traffic)
if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)
opt.model_filename = '{}/flow={}-close={}-period={}-trend={}-spatial={}-mode={}-c={}-s={}-FS={}-model_N={}-scptmodel_d={}-{}-{}-{}'.format(
                    opt.save_dir, opt.flow, opt.close_size,opt.period_size,opt.trend_size,opt.spatial,opt.mode,opt.c,opt.s,opt.FS,opt.model_N,
                    opt.s_model_d,opt.c_model_d,opt.p_model_d,opt.t_model_d)



print('Saving to ' + opt.model_filename)


se = 0

best_model = opt.model_filename+'.model'
print(best_model)
if os.path.exists(best_model):
    saved = torch.load(best_model)
    se = saved['epoch']+1
    opt.best_valid_loss = saved['valid_loss'][-1]
    lr = saved['lr']

lr = 0.001
if opt.lr is not None:
    lr = opt.lr
if opt.se is not None:
    se = opt.se
total_epochs = se + opt.epoch_size


def log(fname, s):
    if not os.path.isdir(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()

def train_epoch(data_type,epoch):
    total_loss = 0
    if data_type == 'train':
        model.train()
        data = train_loader
    if data_type == 'valid':
        model.eval()
        data = valid_loader
    i = 0
    if (opt.period_size > 0) & (opt.close_size > 0) & (opt.trend_size > 0):
        for idx, (c, p, t, target) in enumerate(data):
            #print(idx,c,p,t,target)
            if data_type == 'train':
                scheduler(optimizer,i,epoch)
            optimizer.zero_grad()
            model.zero_grad()
            pred = model(c.float(),opt.mode,opt.c,opt.s,opt.c_t,opt.FS,opt.s_t,opt.flow,p.float(),t.float())
            loss = criterion(pred.float(),target.cuda().float()[:,:,opt.flow])
           
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            i += 1
    elif (opt.period_size > 0) & (opt.close_size > 0):
        for idx, (c, p, target) in enumerate(data):
            #print(idx,c,p,t,target)
            if data_type == 'train':
                scheduler(optimizer,i,epoch)
            optimizer.zero_grad()
            model.zero_grad()
            pred = model(c.float(),opt.mode,opt.c,opt.s,opt.FS,opt.c_t,opt.s_t,opt.flow,p.float())
            loss = criterion(pred.float(), target.cuda().float()[:,:,opt.flow])
            #print(loss)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            i += 1
    return total_loss/len(data)


def train():
    best_valid_loss = opt.best_valid_loss
    train_loss, valid_loss = [], []
    for i in range(se,total_epochs):
        train_loss.append(train_epoch('train',i))
        valid_loss.append(train_epoch('valid',i))

        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]

            torch.save({'epoch': i, 'model': model, 'train_loss': train_loss,
                        'valid_loss': valid_loss,'lr':optimizer.param_groups[0]['lr']}, opt.model_filename + '.model')
            #torch.save(optimizer, opt.model_filename + '.optim')
        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.8f}, valid_loss: {:0.8f}, '
                      'best_valid_loss: {:0.8f}, lr: {:0.8f}').format((i + 1), total_epochs,
                                                                      train_loss[-1],
                                                                      valid_loss[-1],
                                                                      best_valid_loss,
                                                                      optimizer.param_groups[0]['lr'])
        for name, parms in model.named_parameters():
            if parms.grad is not None:
                print('-->name:', name, '-->grad_requirs:', parms.requires_grad)
                print('--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
        print(log_string)
        log(opt.model_filename + '.log', log_string)

def predict(test_type='train'):
    predictions = []
    ground_truth = []
    test = []
    loss = []
    best_model = torch.load(opt.model_filename + '.model').get('model')
    # best_model = model

    if test_type == 'train':
        data = train_loader
    elif test_type == 'test':
        data = test_loader
    elif test_type == 'valid':
        data = valid_loader
    i=0
    if (opt.period_size > 0) & (opt.close_size > 0) & (opt.trend_size > 0):
        for idx, (c, p, t, target) in enumerate(data):
            pred = best_model(c.float(),opt.mode,opt.c,opt.s,opt.c_t,opt.s_t,opt.flow,p.float(),t.float())
            predictions.append(pred.float().data.cpu().numpy())
            ground_truth.append(target.float().numpy()[:,:,opt.flow])
            loss.append(criterion(pred.float(), target.cuda()[:,:,opt.flow]).float().item())
    elif (opt.close_size > 0) & (opt.period_size > 0):
        t = 0
        for idx, (c, p, target) in enumerate(data):
            start = time.time()
            pred = torch.relu(best_model(c.float(),opt.mode,opt.c,opt.s,opt.FS,opt.c_t,opt.s_t,opt.flow,p.float()))
            end = time.time()
            t += (end-start)
            predictions.append(pred.float().data.cpu().numpy())
            ground_truth.append(target.float().numpy()[:,:,opt.flow])
            loss.append(criterion(pred.float(), target.cuda()[:,:,opt.flow]).float().item())
            i += 1
    mrt = t/i
    final_predict = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)
    #plot(final_predict[:,:,10],ground_truth[:,:,10],opt.model_filename)

    factor = mmn.max - mmn.min
    sklearn_mae,sklearn_mse,sklearn_rmse,sklearn_nrmse,sklearn_r2 = getmetrics(final_predict.ravel(),ground_truth.ravel())
    a = mmn.inverse_transform(ground_truth)
    b = mmn.inverse_transform(final_predict)
    mae,mse,rmse,nrmse,r2 = getmetrics(b.ravel(),a.ravel())

    log_string = ' [MSE]:{:0.5f}, [RMSE]:{:0.5f}, [NRMSE]: {:0.5f}, [MAE]:{:0.5f}, [R2]: {:0.5f},[mrt]:{:0.5f}\n'.format(sklearn_mse,sklearn_rmse,sklearn_nrmse,sklearn_mae,sklearn_r2,mrt)+' [Real MSE]:{:0.5f}, [Real RMSE]:{:0.5f}, [Real NRMSE]: {:0.5f}, [Real MAE]:{:0.5f}, [Real R2]: {:0.5f}'.format(mse,rmse,nrmse,mae,r2)
    plot(b[:,:,224],a[:,:,224],opt.model_filename+'real')
    print(log_string)
    print('mean runtime:',mrt)

    log(opt.model_filename + '.log', log_string)
 

def train_valid_split(dataloader, test_size=0.2, shuffle=True, random_seed=0):
    length = len(list(dataloader))
    indices = list(range(0, length))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if type(test_size) is float:
        split = int(np.floor(test_size * length))
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or float'.format(str))
    return indices[split:], indices[:split]


if __name__ == '__main__':
    path = '../all_data_sliced.h5'

    x_train, y_train, x_test, y_test, mmn = load_data(path, opt.traffic, opt.close_size, opt.period_size,
                                                       opt.trend_size,opt.test_size, opt.nb_flow)
    x_train.append(y_train)
    x_test.append(y_test)
    train_data = list(zip(*x_train))
    test_data = list(zip(*x_test))

    # split the training data into train and validation
    train_idx, valid_idx = train_valid_split(train_data,0.1)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
 
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=train_sampler,#num_workers=8,
                               pin_memory=True,drop_last=True)
    valid_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=valid_sampler,#num_workers=8,
                               pin_memory=True,drop_last=True)

    test_loader = DataLoader(test_data, batch_size=opt.test_batch_size, shuffle=False,drop_last=True)

    external_size = 6

    if opt.g is not None:
        GPU = opt.g
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    print("preparing gpu...")
    if torch.cuda.is_available():
        print('using Cuda devices, num:',torch.cuda.device_count())
        print('using GPU:',torch.cuda.current_device())

    if os.path.isfile(best_model):
        #print(best_model)
        model = torch.load(best_model)['model'].cuda()
    else:
        model = T_STGCN(opt.close_size, external_size, opt.model_N, opt.k, opt.spatial,opt.c_model_d,opt.s_model_d,opt.p_model_d,opt.t_model_d).cuda()
    scheduler = LR_Scheduler(opt.lr_scheduler, lr, total_epochs, len(train_loader),warmup_epochs=opt.warmup)
    optimizer = optim.Adam(model.parameters(),lr,betas=(0.9, 0.98), eps=1e-9)

    if not os.path.isdir(opt.save_dir):
        raise Exception('%s is not a dir' % opt.save_dir)

    if opt.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    elif opt.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    print('Training...')
    log(opt.model_filename + '.log', '[training]')
    if opt.train:
        train()
    #predict('train')
    predict('test')
