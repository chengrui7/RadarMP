import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io
import itertools
import time
import math

from Utils import SparsePreprocess

class RadarDataset(Dataset):
    def __init__(self, dataset_dict, params, mode='train', transform=None):
        """
        Args:
            dataset_dict (dict): dict, key-value have file path and label path
            transform (callable, optional): data transform and augmentation
        """
        self.dataset = [frame_data 
                        for seq_dict in dataset_dict.values()
                        for frame_data in seq_dict.values()]
        self.params = params
        self.transform = transform
        self.mode = mode
        self.arrRtensor = torch.tensor(params['radar_FFT_arr']['arrRange'].copy()).to(dtype=torch.float)
        self.arrAtensor = torch.tensor(params['radar_FFT_arr']['arrAzimuth'].copy()).to(dtype=torch.float)
        self.arrEtensor = torch.tensor(params['radar_FFT_arr']['arrElevation'].copy()).to(dtype=torch.float)
        self.arrDtensor = torch.tensor(params['radar_FFT_arr']['arrDoppler'].copy()).to(dtype=torch.float)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        # Transform flow cube
        def transform_and_convert(coords, mask, t_ego_bbx, flows):
            r_vals = coords[0][mask]  # (N,)
            a_vals = coords[1][mask]  # (N,)
            e_vals = coords[2][mask]  # (N,)
            
            x_vals = r_vals * torch.cos(e_vals) * torch.cos(a_vals)
            y_vals = r_vals * torch.cos(e_vals) * torch.sin(a_vals)
            z_vals = r_vals * torch.sin(e_vals)

            ones = torch.ones_like(x_vals)  # (N,)
            points_xyz1 = torch.stack([x_vals, y_vals, z_vals, ones], dim=0)  # (4, N)
            points_transformed = t_ego_bbx @ points_xyz1  # (4, N)
            x_new, y_new, z_new = points_transformed[:3]  # (3, N)

            flows[0][mask] = x_new-x_vals
            flows[1][mask] = y_new-y_vals
            flows[2][mask] = z_new-z_vals
            return flows
        
        def cartesian_to_polar_delta_batch(xyz: torch.Tensor, delta_xyz: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            """
            Convert (dx, dy, dz) to (dr, da, de) for a batch of cartesian points.
            
            Args:
                xyz: Tensor of shape (B, 3, R, A, E) - (x, y, z)
                delta_xyz: Tensor of shape (B, 3, R, A, E) - (dx, dy, dz)
                eps: small value to avoid division by zero
            
            Returns:
                delta_polar: Tensor of shape (B, 3, R, A, E) - (dr, d_azimuth, d_elevation)
            """
            r_vals = xyz[0]  # (N,)
            a_vals = xyz[1]  # (N,)
            e_vals = xyz[2]  # (N,)
            
            x = r_vals * torch.cos(e_vals) * torch.cos(a_vals)
            y = r_vals * torch.cos(e_vals) * torch.sin(a_vals)
            z = r_vals * torch.sin(e_vals)

            dx, dy, dz = delta_xyz[0], delta_xyz[1], delta_xyz[2]
            x2 = x + dx
            y2 = y + dy
            z2 = z + dz

            r1 = torch.sqrt(x**2 + y**2 + z**2 + eps)               # (B, R, A, E)
            a1 = torch.atan2(y, x)                                  # [-pi, pi]
            e1 = torch.atan2(z, torch.sqrt(x**2 + y**2 + eps))      # [-pi/2, pi/2]

            r2 = torch.sqrt(x2**2 + y2**2 + z2**2 + eps)
            a2 = torch.atan2(y2, x2)
            e2 = torch.atan2(z2, torch.sqrt(x2**2 + y2**2 + eps))

            delta_r = r2 - r1
            delta_a = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
            delta_e = (e2 - e1 + math.pi) % (2 * math.pi) - math.pi

            delta_polar = torch.stack([delta_r, delta_a, delta_e], dim=0)  # (3, R, A, E)
            return delta_polar
        
        # Load Radar Flow Cube
        def label2cube(params, label):
            # init
            arrR, arrA, arrE = params['radar_FFT_arr']['arrRange'], \
                        params['radar_FFT_arr']['arrAzimuth'], params['radar_FFT_arr']['arrElevation']
            r_min, r_max, a_min, a_max, e_min, e_max = arrR.min(), arrR.max(), arrA.min(), arrA.max(), arrE.min(), arrE.max()
            r_bin = self.params['radar_FFT_arr']['info_rae'][0][1]
            a_bin = self.params['radar_FFT_arr']['info_rae'][1][1]
            e_bin = self.params['radar_FFT_arr']['info_rae'][2][1]
            bin_tensor = torch.tensor([r_bin, a_bin, e_bin], dtype=torch.float).view(3, 1, 1, 1)
            # rae dimension
            r_dim, a_dim, e_dim = len(arrR), len(arrA), len(arrE)
            r_coords = torch.linspace(r_min, r_max, r_dim).view(r_dim, 1, 1).expand(r_dim, a_dim, e_dim)
            a_coords = torch.linspace(a_min, a_max, a_dim).view(1, a_dim, 1).expand(r_dim, a_dim, e_dim)
            e_coords = torch.linspace(e_min, e_max, e_dim).view(1, 1, e_dim).expand(r_dim, a_dim, e_dim)
            coords = torch.stack([r_coords, a_coords, e_coords], dim=0)
            coords = coords.to(dtype=torch.float)
            # flow calculate
            flows = torch.zeros_like(coords).to(dtype=torch.float)
            mask_bg = torch.full((r_dim, a_dim, e_dim), True, dtype=torch.bool)
            t_rigid = torch.tensor(label['gt_rigid_trans']).to(dtype=torch.float)
            flows = transform_and_convert(coords, mask_bg, t_rigid, flows)
            # fg trans
            for fg_trans in label['gt_fg_rigid_trans']:
                bbox_in_rae, t_ego_bbx, _,_,_,_,_ = fg_trans
                r_minbbox, r_maxbbox, a_minbbox, a_maxbbox, e_minbbox, e_maxbbox = bbox_in_rae
                mask_fg = (coords[0] >= r_minbbox) & (coords[0] <= r_maxbbox) & \
                          (coords[1] >= a_minbbox) & (coords[1] <= a_maxbbox) & \
                          (coords[2] >= e_minbbox) & (coords[2] <= e_maxbbox)
                t_ego_bbx = torch.tensor(t_ego_bbx).to(dtype=torch.float)
                flows = transform_and_convert(coords, mask_fg, t_ego_bbx, flows)
            # coords = pad_cube(coords)
            flows = cartesian_to_polar_delta_batch(coords, flows)
            flows = flows / bin_tensor
            return coords, flows
        
        def pseudolabel(params, label):
            # init
            arrR, arrA, arrE = params['radar_FFT_arr']['arrRange'], \
                        params['radar_FFT_arr']['arrAzimuth'], params['radar_FFT_arr']['arrElevation']
            r_dim, a_dim, e_dim = len(arrR), len(arrA), len(arrE)
            # load data
            pcllist = label['pcllist']
            pcl_label_restored = torch.zeros((r_dim, a_dim, e_dim), dtype=torch.int8)
            indices_tensor = torch.tensor(pcllist)
            pcl_label_restored[indices_tensor[:, 0], indices_tensor[:, 1], indices_tensor[:, 2]] = 1
            negative_tensor = torch.zeros_like(pcl_label_restored)
            pcl_label_restored = torch.where(pcl_label_restored == 1, pcl_label_restored, negative_tensor)
            pcl_label_restored = pcl_label_restored.unsqueeze(0)
            pcl_label = pcl_label_restored.float()
            # load next pcl
            npcllist = label['nextpcllist']
            npcl_label_restored = torch.zeros((r_dim, a_dim, e_dim), dtype=torch.int8)
            nindices_tensor = torch.tensor(npcllist)
            npcl_label_restored[nindices_tensor[:, 0], nindices_tensor[:, 1], nindices_tensor[:, 2]] = 1
            nnegative_tensor = torch.zeros_like(npcl_label_restored)
            npcl_label_restored = torch.where(npcl_label_restored == 1, npcl_label_restored, nnegative_tensor)
            npcl_label_restored = npcl_label_restored.unsqueeze(0)
            npcl_label = npcl_label_restored.float()
            return pcl_label, npcl_label
        
        def load_tensor(params, path, nextpath):
            # load
            cube_tensor = torch.load(path, weights_only=True)
            cube_tensor = (10 ** cube_tensor) 
            nextcube_tensor = torch.load(nextpath, weights_only=True)
            nextcube_tensor = (10 ** nextcube_tensor) 
            # cube tensor normlize
            # global_max = torch.max(cube_tensor.view(-1).max(), nextcube_tensor.view(-1).max())
            # global_min = torch.min(cube_tensor.view(-1).min(), nextcube_tensor.view(-1).min())
            cube_tensor = (cube_tensor - cube_tensor.view(-1).min())/(cube_tensor.view(-1).max() - cube_tensor.view(-1).min())
            nextcube_tensor = (nextcube_tensor - nextcube_tensor.view(-1).min())/(nextcube_tensor.view(-1).max() - nextcube_tensor.view(-1).min())

            return cube_tensor, nextcube_tensor
        
        def orientation_in_AE(ran_arr, azi_arr, ele_arr):
            # create AE direction Vetor
            ran_dim = ran_arr.shape[0]
            azi_dim = azi_arr.shape[0]
            ele_dim = ele_arr.shape[0]
            azi = azi_arr.view(azi_dim, 1)  # [48, 1]
            ele = ele_arr.view(1, ele_dim)   # [1, 16]
            x = torch.cos(ele) * torch.cos(azi)  # [24, 8]
            y = torch.cos(ele) * torch.sin(azi)  
            z = torch.sin(ele).expand(x.shape)   
            direction = torch.stack([x, y, z], dim=0) # [3, 48, 16]
            direction = direction.unsqueeze(1).repeat(1, ran_dim, 1, 1) # [3, 128, 48, 16]
            direction = direction.contiguous()
            return direction
        
        def load_tensor2d(params, path2d, nextpath2d):
            # load
            cube_tensor2d = torch.load(path2d, weights_only=True)
            nextcube_tensor2d = torch.load(nextpath2d, weights_only=True)
            x1ra = cube_tensor2d['tensor_ra'];x1re = cube_tensor2d['tensor_re'];x1ae = cube_tensor2d['tensor_ae']
            x1ra = 10**x1ra;x1ra = (x1ra - x1ra.view(-1).min())/(x1ra.view(-1).max() - x1ra.view(-1).min())
            x1re = 10**x1re;x1re = (x1re - x1re.view(-1).min())/(x1re.view(-1).max() - x1re.view(-1).min())
            x1ae = 10**x1ae;x1ae = (x1ae - x1ae.view(-1).min())/(x1ae.view(-1).max() - x1ae.view(-1).min())
            x1ra = x1ra.unsqueeze(0);x1re = x1re.unsqueeze(0);x1ae = x1ae.unsqueeze(0)
            x2ra = nextcube_tensor2d['tensor_ra'];x2re = nextcube_tensor2d['tensor_re'];x2ae = nextcube_tensor2d['tensor_ae']
            x2ra = 10**x2ra;x2ra = (x2ra - x2ra.view(-1).min())/(x2ra.view(-1).max() - x2ra.view(-1).min())
            x2re = 10**x2re;x2re = (x2re - x2re.view(-1).min())/(x2re.view(-1).max() - x2re.view(-1).min())
            x2ae = 10**x2ae;x2ae = (x2ae - x2ae.view(-1).min())/(x2ae.view(-1).max() - x2ae.view(-1).min())
            x2ra = x2ra.unsqueeze(0);x2re = x2re.unsqueeze(0);x2ae = x2ae.unsqueeze(0)
            return x1ra, x2ra, x1re, x2re, x1ae, x2ae

        # get sample
        sample = self.dataset[idx]
        tesseract_path = sample['tesseract']
        next_tesseract_path = sample['next_tesseract']
        tesseract2d_path = sample['tesseract2d']
        next_tesseract2d_path = sample['next_tesseract2d']
        label_path = sample['label']

        # load data
        if self.mode=='train':
            tesseract, next_tesseract = load_tensor(self.params, tesseract_path, next_tesseract_path)
            if self.transform:
                tesseract = self.transform(tesseract)
                next_tesseract = self.transform(next_tesseract)
            dop = self.arrDtensor
            ori = orientation_in_AE(self.arrRtensor, self.arrAtensor, self.arrEtensor)
            return tesseract, next_tesseract, dop, ori
        elif self.mode=='train2d':
            x1ra, x2ra, x1re, x2re, x1ae, x2ae = load_tensor2d(self.params, tesseract2d_path, next_tesseract2d_path)
            return x1ra, x2ra, x1re, x2re, x1ae, x2ae
        else:
            tesseract, next_tesseract = load_tensor(self.params, tesseract_path, next_tesseract_path)
            if self.transform:
                tesseract = self.transform(tesseract)
                next_tesseract = self.transform(next_tesseract)
            dop = self.arrDtensor
            ori = orientation_in_AE(self.arrRtensor, self.arrAtensor, self.arrEtensor)
            label = np.load(label_path, allow_pickle=True).item()
            coords, flows_label = label2cube(self.params, label)
            pcls_label, npcls_label = pseudolabel(self.params, label)
            return tesseract, next_tesseract, dop, coords, ori, pcls_label, npcls_label, flows_label, sample
        
    