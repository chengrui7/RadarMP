from ruamel.yaml import YAML
params_path = 'Config/params.yaml'
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
from torch.optim.lr_scheduler import MultiStepLR, StepLRfrom, LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.nn.utils import clip_grad_norm_

import scipy.io
import numpy as np
import time
import math

from Utils import ParamLoader, DataRead, IOStream
from Models import RadarMP
from Losses import RadarMPLoss, LossRecord, MGDAWrapper, ValidLoss

def cosine_schedule(epoch, total_epochs, start_value=0.0, end_value=1.0):
    """
    Cosine schedule from start_value to end_value over total_epochs.
    """
    if epoch < total_epochs:
        cos_inner = math.pi * epoch / total_epochs
        return end_value - 0.5 * (end_value - start_value) * (1 + math.cos(cos_inner))
    else: 
        return end_value

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12312'
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size // world_size, sampler=sampler1, prefetch_factor=1, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size // world_size, sampler=sampler2, prefetch_factor=1, num_workers=1)
    # build model
    model = RadarMP.RadarMP(params).to(rank)
    model_path = params.get('path', {}).get('model_path')
    if model_path is not None:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path), strict=False)
    model = DDP(model, device_ids=[rank])
    opt = optim.Adam(model.parameters(), lr=params['train_params']['lr'])
    # scheduler = StepLR(opt, params['train_params']['decay_epochs'], gamma = params['train_params']['decay_rate'])
    lr_lambda = ParamLoader.make_lr_lambda(warmup_epochs=params['train_params']['warmup_epochs'], 
                        decay_epochs=params['train_params']['decay_epochs'], decay_rate=params['train_params']['decay_rate'])  
    scheduler = LambdaLR(opt, lr_lambda)
    # init object
    loss_obj = RadarMPLoss.RadarMPLoss(params)
    val_loss_obj = ValidLoss.ValidLoss(params)
    num_epoch = params['train_params']['epochs']
    num_batch = len(train_loader)
    num_val_batch = len(val_loader)
    loss_record_obj = LossRecord.LossRecord(params, num_batch, num_val_batch, num_epoch, log_dir=f"Checkpoints/{params['exp']['exp_name']}/loss", ios=ios)
    loss_weight_obj = MGDAWrapper.MGDAWrapper(model)
    best_val_loss = np.finfo(np.float32).tiny
    # train epoch
    for epoch in range(params['train_params']['epochs']):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        ios.cprint('----Starting train on the training set----')
        for batch_idx, batch in enumerate(train_loader):
            # take batch
            x1, x2, dop, coords, ori = batch
            x1 = x1.to(rank);x2 = x2.to(rank);dop = dop.to(rank);ori = ori.to(rank)
            # model forward
            seg, flow, x1x2 = model(x1, x2, dop, coords, ori)
            # cal loss
            losses, loss_stem = loss_obj(seg, flow, x1, x2, dop, coords, ori, pcls_label, flows_label, epoch)
            alpha = loss_weight_obj.compute_weighted_loss(losses, x1x2)
            loss = sum(a * l for a, l in zip(alpha, losses))
            loss_stem['total_loss'] = loss
            loss_record_obj.record_batch_loss(loss_stem, alpha, mode='train', batch_idx=batch_idx+1, epoch_idx=epoch+1)
            # backward
            opt.zero_grad()
            loss.backward()
            # clip_grad_norm_(model.parameters(), max_norm=1.0) 
            opt.step()

        loss_record_obj.record_epoch_end(mode='train', epoch_idx=epoch+1)
        # wait other rank
        dist.barrier()

        # validating
        ios.cprint('----Starting evaluation on the validation set----')
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # take batch
                x1, x2, dop, coords, ori, pcls_label, _, flows_label, _ = batch
                x1 = x1.to(rank); x2 = x2.to(rank); dop = dop.to(rank); coords = coords.to(rank); ori = ori.to(rank)
                pcls_label = pcls_label.to(rank); flows_label = flows_label.to(rank)
                seg, flow, x1x2 = model(x1, x2, dop, coords, ori)
                # cal loss
                loss, loss_stem = val_loss_obj(seg, flow, pcls_label, flows_label)
                loss_record_obj.record_batch_loss(loss_stem, weight=None, mode='val', batch_idx=batch_idx+1, epoch_idx=epoch+1)
                val_loss += loss.item() * x1.shape[0]
            # wait other rank
            dist.barrier()
            # cal avg loss
            loss_record_obj.record_epoch_end(mode='val', epoch_idx=epoch+1)
            # save best model
            val_loss /= len(val_loader.dataset)
            if rank==0 : 
                torch.save(model.module.state_dict(), f"Checkpoints/{params['exp']['exp_name']}/models/model_{epoch}.t7")
            ios.cprint(f"Save model in checkpoints/{params['exp']['exp_name']}/models/model_{epoch}.t7  .....")
        
        scheduler.step()

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
    
    os.system('cp train.py Checkpoints' + '/' + params['exp']['exp_name'] + '/' + 'train.py.backup')
    os.system('cp Config/params.yaml Checkpoints' + '/' + params['exp']['exp_name'] + '/' + 'train_params.yaml.backup')


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
    if params['exp']['mode'] == 'train':
        world_size = torch.cuda.device_count() 
        train_dataset = DataRead.RadarDataset(trian_dict, params, mode='train')
        val_dataset = DataRead.RadarDataset(val_dict, params, mode='val')
        mp.spawn(train, args=(world_size, params, train_dataset, val_dataset, batch_size), nprocs=world_size, join=True)
    else:
        raise ValueError(f"Invalid Train Mode: {params['exp']['mode']}")

    


