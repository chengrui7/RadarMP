import os
import torch
torch.cuda.empty_cache()
import torch.optim as optim
import torch.nn as nn

import scipy.io
import numpy as np
import time

from utils import Datadict
from utils import Dataloader
from utils import Annoutil

if __name__ == '__main__':
    arr_range, arr_azimuth, arr_elevation, arr_doppler = Annoutil.load_physical_values()
    coords = Annoutil.coordinate_generate(arr_range, arr_azimuth, arr_elevation)
    arrR_tensor = torch.tensor(arr_range.copy()).to(dtype=torch.float)
    arrA_tensor = torch.tensor(arr_azimuth.copy()).to(dtype=torch.float)
    arrE_tensor = torch.tensor(arr_elevation.copy()).to(dtype=torch.float)
    arrD_tensor = torch.tensor(arr_doppler.copy()).to(dtype=torch.float)
    # eval seq
    seq = ['5', '42', '21', '35']  # '35','5', '1','30','21', '7', '16', '44', '42'
    lidarroi = {'x':[0, 60.0],'y': [-50.0, 50.0],'z': [-30, 30]}
    snr_metric = []
    pfa_metric = []
    pd_metric = []
    cd_metric = []
    # load other params
    for eseq in seq:
        # path setting
        head_path = '/home/data/K-RadarTemp/K-RadarTemp/' + eseq + '/'
        cube_path = '/home/data/K-Radar/K-Radar/generated_files/' + eseq + '/radar_tensor'
        rpcl_path = '/home/data/K-RadarTemp/K-RadarTemp/' + eseq + '/rpdnet'
        rocc_path = '/home/data/K-RadarTemp/K-RadarTemp/' + eseq + '/rpdnetocc'
        lpcl_path = '/home/data/K-RadarTemp/K-RadarTemp/' + eseq + '/os2-64'
        # load file
        lpcl_list = sorted([f for f in os.listdir(lpcl_path) if f.endswith('.pcd')])
        cube_list = sorted([f for f in os.listdir(cube_path) if f.endswith('.pt')])
        rpcl_list = sorted([f for f in os.listdir(rpcl_path) if f.endswith('.bin')])
        rocc_list = sorted([f for f in os.listdir(rocc_path) if f.endswith('.npy')])
        # param setting 
        calib_path = head_path + 'info_calib/calib_radar_lidar.txt'
        f = open(calib_path, 'r')
        lines = f.readlines()
        f.close()
        list_calib = list(map(lambda x: float(x), lines[1].split(',')))
        frame_dif = int(list_calib[0])
        calib = np.array([list_calib[1], list_calib[2], 0.7])
        # cal metric
        for i in range(len(os.listdir(rpcl_path))):
            # file path
            lpcl_file = os.path.join(lpcl_path, lpcl_list[i])
            rpcl_file = os.path.join(rpcl_path, rpcl_list[i])
            rocc_file = os.path.join(rocc_path, rocc_list[i])
            cube_file = os.path.join(cube_path, cube_list[i])
            # load file
            lidar_pcd = Dataloader.PointCloudPcd(lpcl_file)
            lidar_pcd.calib_xyz([-2.54,0.3,0.7])
            lidar_pcd.roi_filter(lidarroi)
            label_seg = Annoutil.check_points_within_radius_kdtree(coords, lidar_pcd.points)
            rocc = np.load(rocc_file)
            rocc = rocc.reshape(-1, 4)
            cube = torch.load(cube_file, weights_only=True)
            # cube = (10 ** cube) / 1e+13
            dop, ran, azi, ele = cube.shape
            rpcl = np.fromfile(rpcl_file, dtype=np.float32).reshape(-1, 7)
            # cal pfa and pd
            seg = torch.zeros([ran, azi, ele])
            seg[rocc[:,1],rocc[:,2],rocc[:,3]] = 1
            seg = seg > 0.9
            label_seg = label_seg.squeeze(0)
            label_seg = label_seg > 0.9
            TP = (seg & label_seg).sum().item()
            FP = (seg & (~label_seg)).sum().item()
            FN = ((~seg) & label_seg).sum().item()
            TN = ((~seg) & (~label_seg)).sum().item()
            Pfa = FP / (FP + TN + 1e-6)  # sum a epsilon
            Pd = TP / (TP + FN + 1e-6)
            pfa_metric.append(Pfa)
            pd_metric.append(Pd)
            # cal chamfer 
            chamdist = Annoutil.chamfer_distance(lidar_pcd.roipoints, rpcl[:,:3])
            cd_metric.append(chamdist)
            # cal SNR
            rocc = torch.from_numpy(rocc).long()
            d_idx, r_idx, a_idx, e_idx = rocc[:, 0], rocc[:, 1], rocc[:, 2], rocc[:, 3]
            signal = cube[d_idx, r_idx, a_idx, e_idx]
            mask = torch.ones((dop, ran, azi, ele), dtype=torch.bool, device=cube.device)
            mask[d_idx, r_idx, a_idx, e_idx] = False
            noise = cube[mask]
            SNR = 10 * (signal.mean() - noise.mean())
            snr_metric.append(SNR.item())
            print(f'One time Metric: cd:{chamdist}, snr:{SNR.item()}, pd:{Pd}, pfa:{Pfa}')
    # build array
    cd_metric = np.stack(cd_metric)
    snr_metric = np.stack(snr_metric)
    pd_metric = np.stack(pd_metric)
    pfa_metric = np.stack(pfa_metric)
    # cal mean
    cd_metric = np.mean(cd_metric)
    snr_metric = np.mean(snr_metric)
    pd_metric = np.mean(pd_metric)
    pfa_metric = np.mean(pfa_metric)
    print(f'Metric: cd:{cd_metric}, snr:{snr_metric}, pd:{pd_metric}, pfa:{pfa_metric}')

            


