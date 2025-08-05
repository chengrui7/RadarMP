from ruamel.yaml import YAML
params_path = 'Config/evalparams.yaml'
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
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.nn.utils import clip_grad_norm_

import scipy.io
import numpy as np
import time

from Utils import ParamLoader, DataRead, IOStream
from Models import RadarMP
from Losses import ValidLoss, ComputeMetric

def eval_model(params, model, test_loader):
    # init
    ios = IOStream.IOStream(f"Checkpoints/{params['exp']['exp_name']}/log/eval.log")
    ios.cprint(f"---- '{params['exp']['exp_name']} vis' ----")
    # eval
    model.eval()
    with torch.no_grad():
        cm_obj = ComputeMetric.ComputeMetric(params)
        for batch_idx, batch in enumerate(test_loader):
            # take batch
            x1, x2, dop, coords, ori, pcls_label, npcls_label, flows_label, sample_dict = batch
            x1 = x1.to('cuda'); x2 = x2.to('cuda'); dop = dop.to('cuda'); coords = coords.to('cuda'); ori = ori.to('cuda')
            pcls_label = pcls_label.to('cuda'); flows_label = flows_label.to('cuda')
            seg, flow = model(x1, x2, dop, coords, ori)
            # cal loss
            snr, cd, pd, pfa, epe, rne, accs, accr = cm_obj.compute_pointflow_metric(x1, seg, flow, pcls_label, flows_label)
            ios.cprint(f'Batch {batch_idx}: snr={snr}, cd={cd}, pd={pd}, pfa={pfa}'
                       f'epe={epe}, rne={rne}, accs={accs}, accr={accr}')
        snrm, cd_m, pdm, pfam, epe_m, rne_m, accs_m, accr_m = cm_obj.output_metric()
        ios.cprint(f'Eval Metrics: snr={snrm}, cd={cd_m}, pd={pdm}, pfa={pfam}'
                   f'epe={epe_m}, rne={rne_m}, accs={accs_m}, accr={accr_m}')

def _init_(params):

    if not os.path.exists('Checkpoints/' + params['exp']['exp_name'] + '/' + 'log'):
        os.makedirs('Checkpoints/' + params['exp']['exp_name'] + '/' + 'log')
    
    os.system('cp eval.py Checkpoints' + '/' + params['exp']['exp_name'] + '/' + 'eval.py.backup')
    os.system('cp Config/evalparams.yaml Checkpoints' + '/' + params['exp']['exp_name'] + '/' + 'eval_params.yaml.backup')


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
    test_dict = ParamLoader.load_radarcube(params['path']['data_path'], params['exp']['seq'], params['path']['radartensor_path'],
                    params['path']['radartensor2d_path'], params['path']['label_path'], params['path']['calib_path'], params['path']['lidar_path'])

    # # load model build dataLoader
    if params['exp']['mode'] == 'eval':
        # load data
        test_dataset = DataRead.RadarDataset(test_dict, params, mode='val')
        test_loader = DataLoader(test_dataset, batch_size=params['eval_params']['batch_size'], shuffle=False, prefetch_factor=1, num_workers=1)
        # load eval model
        model_path = 'Checkpoints/' + params['exp']['exp_name'] + '/models/' + params['path']['model_path']
        if not os.path.exists(model_path):
            raise ValueError(f"Can't find trained model: {model_path}")
        model = RadarMP.RadarMP(params)
        model = model.to('cuda')
        model.load_state_dict(torch.load(model_path), strict=False)
        eval_model(params, model, test_loader)
    else:
        raise ValueError(f"Invalid Mode at eval: {params['exp']['mode']}")