from ruamel.yaml import YAML
params_path = 'Config/params2dflow.yaml'
# load Yaml
yaml_obj = YAML()
yaml_obj.preserve_quotes = True
with open(params_path, 'r') as file:
    params = yaml_obj.load(file)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = params['exp']['cuda_device']
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
torch.cuda.empty_cache()
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import MultiStepLR, StepLR, LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.nn.utils import clip_grad_norm_

import scipy.io
import numpy as np
import time

from Utils import ParamLoader, DataRead, IOStream
from Models import Motion2DPerception
from Losses import RadarMPLoss2d, LossRecord

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, params, train_dataset, val_dataset, batch_size):

    setup(rank, world_size)
    ios = IOStream.IOStream(f"Checkpoints/{params['exp']['exp_name']}/log/train_{rank}.log")
    ios.cprint(f"---- '{params['exp']['exp_name']} train' ----")
    # load sampler data
    sampler1 = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    sampler2 = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size // world_size, sampler=sampler1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size // world_size, sampler=sampler2)
    # build model
    model = Motion2DPerception.Motion2DPerception().to(rank)
    model = DDP(model, device_ids=[rank])
    opt = optim.Adam(model.parameters(), lr=params['train_params']['lr'])
    #lr_lambda = ParamLoader.make_lr_lambda(warmup_epochs=params['train_params']['warmup_epochs'], 
                        # decay_epochs=params['train_params']['decay_epochs'], decay_rate=params['train_params']['decay_rate'])  
    # scheduler = LambdaLR(opt, lr_lambda)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.85, patience=3, verbose=True, 
                                        threshold=0.001, threshold_mode='abs', cooldown=1, min_lr=1e-6)
    # init object
    loss_obj = RadarMPLoss2d.PowerFlow2DLoss(params)
    num_epoch = params['train_params']['epochs']
    num_batch = len(train_loader)
    num_val_batch = len(val_loader)
    loss_record_obj = LossRecord.LossRecord(params, num_batch, num_val_batch, num_epoch, log_dir=f"Checkpoints/{params['exp']['exp_name']}/loss", ios=ios)
    best_val_loss = np.inf
    # train epoch
    for epoch in range(params['train_params']['epochs']):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        ios.cprint('----Starting train on the training set----')
        for batch_idx, batch in enumerate(train_loader):
            # take batch
            x1ra, x2ra, x1re, x2re, x1ae, x2ae = batch
            x1ra = x1ra.to(rank);x1re = x1re.to(rank);x1ae = x1ae.to(rank)
            x2ra = x2ra.to(rank);x2re = x2re.to(rank);x2ae = x2ae.to(rank)
            ios.cprint(f'max-mean-min RA: {x1ra.max()},{x1ra.mean()},{x1ra.min()}, RE: {x1re.max()},{x1re.mean()},{x1re.min()}, AE: {x1ae.max()},{x1ae.mean()},{x1ae.min()}')
            # model forward
            flow_ra, flow_re, flow_ae = model(x1ra, x2ra, x1re, x2re, x1ae, x2ae)
            # cal loss
            loss, loss_stem = loss_obj(flow_ra, flow_re, flow_ae, x1ra, x2ra, x1re, x2re, x1ae, x2ae)
            loss_record_obj.record_batch_loss(loss_stem, weight=None, mode='train', batch_idx=batch_idx+1, epoch_idx=epoch+1)
            # backward
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0) 
            opt.step()

        loss_record_obj.record_epoch_end(mode='train', epoch_idx=epoch+1)
        # wait other rank
        dist.barrier()

        # validating
        ios.cprint('----Starting evaluation on the validation set----')
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # take batch
                x1ra, x2ra, x1re, x2re, x1ae, x2ae = batch
                x1ra = x1ra.to(rank);x1re = x1re.to(rank);x1ae = x1ae.to(rank)
                x2ra = x2ra.to(rank);x2re = x2re.to(rank);x2ae = x2ae.to(rank)
                flow_ra, flow_re, flow_ae = model(x1ra, x2ra, x1re, x2re, x1ae, x2ae)
                # cal loss
                loss, loss_stem = loss_obj(flow_ra, flow_re, flow_ae, x1ra, x2ra, x1re, x2re, x1ae, x2ae)
                loss_record_obj.record_batch_loss(loss_stem, weight=None, mode='val', batch_idx=batch_idx+1, epoch_idx=epoch+1)
            # wait other rank
            dist.barrier()
            # cal avg loss
            epoch_avg_valloss = loss_record_obj.record_epoch_end(mode='val', epoch_idx=epoch+1)
            epoch_avg_valloss = next(iter(epoch_avg_valloss.values()))
            # save best model
            if rank==0 and epoch_avg_valloss < best_val_loss:
                torch.save(model.module.state_dict(), f"Checkpoints/{params['exp']['exp_name']}/models/flow2dmodel_{epoch+1}.t7")
                best_val_loss = epoch_avg_valloss
                ios.cprint(f"Save best model in checkpoints/{params['exp']['exp_name']}/models/flow2dmodel_{epoch+1}.t7  .....")
        
        scheduler.step(epoch_avg_valloss)

    cleanup()
    ios.cprint(f"----{rank}: Finishing training and validating----")

def _init_(params):
    if not os.path.exists('Checkpoints'):
        os.makedirs('Checkpoints')
    if not os.path.exists('Checkpoints/' + params['exp']['exp_name']):
        os.makedirs('Checkpoints/' + params['exp']['exp_name'])
    if not os.path.exists('Checkpoints/' + params['exp']['exp_name'] + '/' + 'models'):
        os.makedirs('Checkpoints/' + params['exp']['exp_name'] + '/' + 'models')
    if not os.path.exists('Checkpoints/' + params['exp']['exp_name'] + '/' + 'loss'):
        os.makedirs('Checkpoints/' + params['exp']['exp_name'] + '/' + 'loss')
    if not os.path.exists('Checkpoints/' + params['exp']['exp_name'] + '/' + 'log'):
        os.makedirs('Checkpoints/' + params['exp']['exp_name'] + '/' + 'log')
    
    os.system('cp train_flow2d.py Checkpoints' + '/' + params['exp']['exp_name'] + '/' + 'train.py.backup')
    os.system('cp Config/params2dflow.yaml Checkpoints' + '/' + params['exp']['exp_name'] + '/' + 'train_params.yaml.backup')


if __name__ == '__main__':

    # load other params
    arrR, arrA, arrE, arrD = ParamLoader.load_axis(params['path']['path_head'], params['path']['axis_path'])
    params['radar_FFT_arr'] = {} 
    params['radar_FFT_arr']['arrRange'] = arrR
    params['radar_FFT_arr']['arrAzimuth'] = arrA
    params['radar_FFT_arr']['arrElevation'] = arrE
    params['radar_FFT_arr']['arrDoppler'] = arrD
    r_min, r_max, a_min, a_max, e_min, e_max = arrR.min(), arrR.max(), arrA.min(), arrA.max(), arrE.min(), arrE.max()
    r_bin, a_bin, e_bin = np.mean(arrR[1:]-arrR[:-1]), np.mean(arrA[1:]-arrA[:-1]), np.mean(arrE[1:]-arrE[:-1])
    params['radar_FFT_arr']['info_rae'] = [[r_min, r_bin, r_max], [a_min, a_bin, a_max], [e_min, e_bin, e_max]]

    # init
    _init_(params)
    torch.cuda.manual_seed_all(params['exp']['seed'])
    np.random.seed(params['exp']['seed'])
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(params['exp']['seed'])

    # split dataset
    trian_dict = ParamLoader.load_radarcube(params['path']['data_path'], params['exp']['train_seq'], params['path']['radartensor_path'],
                    params['path']['radartensor2d_path'], params['path']['label_path'], params['path']['calib_path'], params['path']['lidar_path'])
    val_dict = ParamLoader.load_radarcube(params['path']['data_path'], params['exp']['val_seq'], params['path']['radartensor_path'],
                    params['path']['radartensor2d_path'], params['path']['label_path'], params['path']['calib_path'], params['path']['lidar_path'])

    batch_size = params['train_params']['batch_size']
    # # load model build dataLoader
    if params['exp']['mode'] == 'train2d':
        world_size = torch.cuda.device_count() 
        train_dataset = DataRead.RadarDataset(trian_dict, params, mode=params['exp']['mode'])
        val_dataset = DataRead.RadarDataset(val_dict, params, mode=params['exp']['mode'])
        mp.spawn(train, args=(world_size, params, train_dataset, val_dataset, batch_size), nprocs=world_size, join=True)
        # train_Parallel(params, train_dataset, val_dataset, batch_size)
    else:
        raise ValueError(f"Invalid Train Mode: {params['exp']['mode']}")