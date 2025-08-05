import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import math
import matplotlib.pyplot as plt

import numpy as np

class RadarMPLoss(nn.Module):
    def __init__(self, params):
        super(RadarMPLoss, self).__init__()
        self.params = params
        self.pfl = PowerFlowLoss(params)
        self.spl = SegPowerLoss(params)
        self.sdl = SegDopplerLoss(params)
        self.sfl = SegFlowConsistencyLoss(params)

    def compute_topk_mask_per_r(self, x1, k=5, r_start=10):
        B, C, R, A, E = x1.shape
        # init
        x1_rae = torch.mean(x1, dim=1, keepdim=True)  # or use max
        pseudo_mask = torch.zeros_like(x1_rae)
        # for r=10 to R-1
        for r in range(r_start, R):
            # r slice: shape (B, 1, A, E)
            x_r = x1_rae[:, :, r, :, :]   # shape: (B,1,A,E)
            x_r_flat = x_r.view(B, -1)
            # topk
            topk_vals, topk_idx = torch.topk(x_r_flat, k=k, dim=1)
            # mask: (B, A*E)
            mask_r_flat = torch.zeros_like(x_r_flat)
            mask_r_flat.scatter_(1, topk_idx, 1.0)
            # reshape (B,1,A,E)
            mask_r = mask_r_flat.view(B, 1, A, E)
            pseudo_mask[:, :, r, :, :] = mask_r
        return pseudo_mask
        

    def forward(self, seg, flow, x1, x2, dop, coords, ori, pcllabel, flowlabel, epoch):

        pfl_ = self.pfl(flow, x1, x2)
        spl_ = self.spl(seg, x1)
        sdl_ = self.sdl(seg, flow, x1, dop, coords, ori)
        sfl_ = self.sfl(seg, flow, x2)
        # ral_ = (seg.sigmoid()).mean()

        loss_item = {
             'segpower_loss': spl_,
             'segdop_loss': sdl_,
             'powerflow_loss': pfl_,
             # 'recall_loss': ral_,
             'segflow_loss': sfl_,
        }
        return [spl_, pfl_, sdl_, sfl_], loss_item 

class PowerFlowLoss(nn.Module):
    def __init__(self, params):
        super(PowerFlowLoss, self).__init__()
        self.params = params
        self.loss_fn = nn.MSELoss(reduction='none')
    def warp_3d(self, x, flo):
        B, C, D, H, W = x.shape
        flo = flo.permute(0, 2, 1).contiguous().view(B, 3, D, H, W)
        # 1. Create mesh grid
        zz = torch.arange(0, D).view(-1, 1, 1).repeat(1, H, W)  # [D, H, W]
        yy = torch.arange(0, H).view(1, -1, 1).repeat(D, 1, W)  # [D, H, W]
        xx = torch.arange(0, W).view(1, 1, -1).repeat(D, H, 1)  # [D, H, W]
        zz = zz.view(1, 1, D, H, W).repeat(B, 1, 1, 1, 1)  # [B, 1, D, H, W]
        yy = yy.view(1, 1, D, H, W).repeat(B, 1, 1, 1, 1)
        xx = xx.view(1, 1, D, H, W).repeat(B, 1, 1, 1, 1)
        grid = torch.cat((xx, yy, zz), dim=1).float()  # [B, 3, D, H, W]
        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo  # [B, 3, D, H, W]
        # 2. Scale to [-1, 1]
        vgrid_norm = torch.zeros_like(vgrid)
        vgrid_norm[:, 0, :, :, :] = 2.0 * vgrid[:, 0, :, :, :] / (W - 1) - 1.0  # x
        vgrid_norm[:, 1, :, :, :] = 2.0 * vgrid[:, 1, :, :, :] / (H - 1) - 1.0  # y
        vgrid_norm[:, 2, :, :, :] = 2.0 * vgrid[:, 2, :, :, :] / (D - 1) - 1.0  # z
        # 3. Permute to [B, D, H, W, 3] for grid_sample
        vgrid_norm = vgrid_norm.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)
        # 4. Sample
        output = F.grid_sample(x, vgrid_norm, mode='bilinear', padding_mode='border', align_corners=True)
        return output
    def forward(self, flow, x1, x2):
        x1_rae = torch.mean(x1, dim=1, keepdim=True); x2_rae = torch.mean(x2, dim=1, keepdim=True)
        max_x1 = F.gumbel_softmax(x1, tau=1.0, hard=True, dim=1); max_x2 = F.gumbel_softmax(x2, tau=1.0, hard=True, dim=1)
        x1_dopmax = torch.sum(max_x1 * x1, dim=1, keepdim=True); x2_dopmax = torch.sum(max_x2 * x2, dim=1, keepdim=True)
        x1_pow = torch.cat([x1_rae, x1_dopmax], dim=1); x2_pow = torch.cat([x2_rae, x2_dopmax], dim=1)
        # reduction ran azi ele
        x2_warped = self.warp_3d(x2_pow, flow)
        loss = x1_pow * self.loss_fn(x1_pow, x2_warped)
        loss = loss.mean()
        return loss
    

class SegPowerLoss(nn.Module):
    def __init__(self, params):
        super(SegPowerLoss, self).__init__()
        self.params = params
        self.loss_fn = nn.L1Loss()
        
    def forward(self, seg, x1):
        # reduction dop
         # reduction dop
        bs, dop, ran, azi, ele = x1.shape
        max_x1 = F.gumbel_softmax(x1, tau=1.0, hard=True, dim=1) 
        x1_doppow1 = torch.sum(max_x1 * x1, dim=1, keepdim=True)
        x1_rae = torch.sum(x1, dim=1, keepdim=True)
        x1_pow = torch.cat([x1_doppow1, x1_rae], dim=1)
        # process seg
        x1_avg_pow = F.avg_pool3d(x1_pow, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        seg_label = torch.sigmoid(x1_pow-1.01*x1_avg_pow)
        seg_label = torch.mean(seg_label, dim=1, keepdim=True)
        seg = seg.permute(0,2,1).contiguous().view(bs, -1, ran, azi, ele)
        seg = seg.sigmoid()
        loss = self.loss_fn(seg, seg_label)
        return loss

class SegDopplerLoss(nn.Module):
    def __init__(self, params):
        super(SegDopplerLoss, self).__init__()
        self.params = params
        self.interval = 0.1
        self.loss_fn = nn.L1Loss()

    def warp_flowxyz(self, coords, flo):
        B, C, D, H, W = coords.shape
        # 1. Create mesh grid
        zz = torch.arange(0, D).view(-1, 1, 1).repeat(1, H, W)  # [D, H, W]
        yy = torch.arange(0, H).view(1, -1, 1).repeat(D, 1, W)  # [D, H, W]
        xx = torch.arange(0, W).view(1, 1, -1).repeat(D, H, 1)  # [D, H, W]
        zz = zz.view(1, 1, D, H, W).repeat(B, 1, 1, 1, 1)  # [B, 1, D, H, W]
        yy = yy.view(1, 1, D, H, W).repeat(B, 1, 1, 1, 1)
        xx = xx.view(1, 1, D, H, W).repeat(B, 1, 1, 1, 1)
        grid = torch.cat((xx, yy, zz), dim=1).float()  # [B, 3, D, H, W]
        if coords.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo  # [B, 3, D, H, W]
        # 2. Scale to [-1, 1]
        vgrid_norm = torch.zeros_like(vgrid)
        vgrid_norm[:, 0, :, :, :] = 2.0 * vgrid[:, 0, :, :, :] / (W - 1) - 1.0  # x
        vgrid_norm[:, 1, :, :, :] = 2.0 * vgrid[:, 1, :, :, :] / (H - 1) - 1.0  # y
        vgrid_norm[:, 2, :, :, :] = 2.0 * vgrid[:, 2, :, :, :] / (D - 1) - 1.0  # z
        # 3. Permute to [B, D, H, W, 3] for grid_sample
        vgrid_norm = vgrid_norm.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)
        # 4. trans polar to cartian
        r_vals = coords[:, 0, :, :, :]
        a_vals = coords[:, 1, :, :, :]
        e_vals = coords[:, 2, :, :, :]
        x_vals = r_vals * torch.cos(e_vals) * torch.cos(a_vals)
        y_vals = r_vals * torch.cos(e_vals) * torch.sin(a_vals)
        z_vals = r_vals * torch.sin(e_vals)
        xyzcoords = torch.stack([x_vals, y_vals, z_vals], dim=1)
        # 4. Sample
        xyznewcoords = F.grid_sample(xyzcoords, vgrid_norm, mode='bilinear', padding_mode='border', align_corners=True)
        output = xyznewcoords - xyzcoords
        return output

    def forward(self, seg, flow, x1, dop_arr, coords, ori):
        # reduction dop
        bs, dop, ran, azi, ele = x1.shape
        dop_arr = dop_arr.view(bs, dop, 1, 1, 1)
        max_x1 = F.gumbel_softmax(x1, tau=1.0, hard=True, dim=1) 
        softmax_x1 = F.softmax(x1, dim=1) 
        x1_dopvalue1 = torch.sum(max_x1 * dop_arr, dim=1, keepdim=True)
        x1_dopvalue2 = torch.sum(softmax_x1 * dop_arr, dim=1, keepdim=True)
        x1_dop = torch.cat([x1_dopvalue1, x1_dopvalue2], dim=1)
        # process seg and flow
        seg = seg.permute(0,2,1).contiguous().view(bs, -1, ran, azi, ele)
        flow = flow.permute(0,2,1).contiguous().view(bs, -1, ran, azi, ele)
        flow_xyz = self.warp_flowxyz(coords, flow)
        velo = flow_xyz / self.interval # B N 3
        dop_label = torch.sum(velo * ori, dim=1, keepdim=True)
        dop_error = (dop_label - x1_dop) ** 2 
        seg_label = 10.0 * (0.15 - dop_error)  
        seg_label = torch.sigmoid(seg_label)  # 0.15 mid point（0.5） 10.0 sharpness bigger to sharpper
        seg_label = torch.mean(seg_label, dim=1, keepdim=True)
        seg = seg.sigmoid()
        loss = self.loss_fn(seg, seg_label)
        #intersection = (seg.sigmoid() * seg_label).sum()
        #dice = 1 - (2 * intersection + 1e-5) / (seg.pow(2).sum() + seg_label.pow(2).sum() + 1e-5)
        return loss # + dice

class SegFlowConsistencyLoss(nn.Module):
    def __init__(self, params):
        super(SegFlowConsistencyLoss, self).__init__()
        self.params = params
        self.loss_fn = nn.L1Loss()
    def warp_3d(self, x, flo):
        B, C, D, H, W = x.shape
        flo = flo.permute(0, 2, 1).contiguous().view(B, 3, D, H, W)
        # 1. Create mesh grid
        zz = torch.arange(0, D).view(-1, 1, 1).repeat(1, H, W)  # [D, H, W]
        yy = torch.arange(0, H).view(1, -1, 1).repeat(D, 1, W)  # [D, H, W]
        xx = torch.arange(0, W).view(1, 1, -1).repeat(D, H, 1)  # [D, H, W]
        zz = zz.view(1, 1, D, H, W).repeat(B, 1, 1, 1, 1)  # [B, 1, D, H, W]
        yy = yy.view(1, 1, D, H, W).repeat(B, 1, 1, 1, 1)
        xx = xx.view(1, 1, D, H, W).repeat(B, 1, 1, 1, 1)
        grid = torch.cat((xx, yy, zz), dim=1).float()  # [B, 3, D, H, W]
        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo  # [B, 3, D, H, W]
        # 2. Scale to [-1, 1]
        vgrid_norm = torch.zeros_like(vgrid)
        vgrid_norm[:, 0, :, :, :] = 2.0 * vgrid[:, 0, :, :, :] / (W - 1) - 1.0  # x
        vgrid_norm[:, 1, :, :, :] = 2.0 * vgrid[:, 1, :, :, :] / (H - 1) - 1.0  # y
        vgrid_norm[:, 2, :, :, :] = 2.0 * vgrid[:, 2, :, :, :] / (D - 1) - 1.0  # z
        # 3. Permute to [B, D, H, W, 3] for grid_sample
        vgrid_norm = vgrid_norm.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)
        # 4. Sample
        output = F.grid_sample(x, vgrid_norm, mode='bilinear', padding_mode='border', align_corners=True)
        return output
    def forward(self, seg, flow, x2):
        bs, dop, ran, azi, ele = x2.shape
        x2_rae = torch.mean(x2, dim=1, keepdim=True)
        # reduction ran azi ele
        x2_warped = self.warp_3d(x2_rae, flow)
        x2_avg_pow = F.avg_pool3d(x2_warped, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        seg_label = torch.sigmoid(x2_warped-1.01*x2_avg_pow)
        seg_label = torch.mean(seg_label, dim=1, keepdim=True)
        seg = seg.permute(0,2,1).contiguous().view(bs, -1, ran, azi, ele)
        seg = seg.sigmoid()
        loss = self.loss_fn(seg, seg_label)
        return loss   

class ChamferDistanceLoss(nn.Module):

    def __init__(self, params):
        super(ChamferDistanceLoss, self).__init__()
        self.params = params
        self.loss_fn = nn.BCELoss()
        arrR, arrA, arrE = params['radar_FFT_arr']['arrRange'], params['radar_FFT_arr']['arrAzimuth'], params['radar_FFT_arr']['arrElevation']
        arrR, arrA, arrE = torch.tensor(arrR).to(dtype=torch.float), torch.tensor(arrA).to(dtype=torch.float), torch.tensor(arrE).to(dtype=torch.float)
        self.r_dim, self.a_dim, self.e_dim = len(arrR), len(arrA), len(arrE)

    def forward(self, pcl1, pcl2, flow):

        # create normalize grid index
        device = pcl1.device
        b = pcl1.shape[0]
        r = torch.linspace(-1, 1, self.r_dim)
        a = torch.linspace(-1, 1, self.a_dim)
        e = torch.linspace(-1, 1, self.e_dim)
        grid_r, grid_a, grid_e = torch.meshgrid(r, a, e, indexing='ij')
        grid = torch.stack((grid_e, grid_a, grid_r), dim=-1).to(device) # Note: grid_sample uses (x,y,z) = (e,a,r) order
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1, 1)
        flow_r = flow[:, 0] / (self.r_dim - 1) * 2  # dr in R dimension
        flow_a = flow[:, 1] / (self.a_dim - 1) * 2  # da in A dimension
        flow_e = flow[:, 2] / (self.e_dim - 1) * 2  # de in E dimension
        flow_grid = torch.stack((flow_r, flow_a, flow_e), dim=-1)  # [B, R, A, E, 3]
        warped_grid = grid + flow_grid
        # preprocess
        pcl1 = F.softmax(pcl1, dim=1)
        pcl1_object = pcl1[:,1,:,:,:]
        pcl1_object = pcl1_object.unsqueeze(1) # B * 1 * R * A * E
        pcl2 = F.softmax(pcl2, dim=1)
        pcl2_object = pcl2[:,1,:,:,:]
        pcl2_object = pcl2_object.unsqueeze(1)
        # cal pcl loss
        warped_pcl1_object = F.grid_sample(pcl2_object, warped_grid, mode='bilinear', padding_mode='border',align_corners=True)
        loss = self.loss_fn(pcl1_object, warped_pcl1_object)

        return loss
    
class ExistLoss(nn.Module):
    def __init__(self, params, target_mean=3e-3):
        super(ExistLoss, self).__init__()
        self.params = params
        self.target_mean = target_mean
        self.loss_fn = nn.MSELoss()
    
    def forward(self, pcl1, pcl2):
        # preprocess
        pcl1 = F.softmax(pcl1, dim=1)
        pcl2 = F.softmax(pcl2, dim=1)
        pcl1_object = pcl1[:,1,:,:,:]
        pcl1_object = pcl1_object.unsqueeze(1) # B * 1 * R * A * E
        pcl2_object = pcl2[:,1,:,:,:]
        pcl2_object = pcl2_object.unsqueeze(1)
        # sum
        loss1 = torch.log(pcl1_object.mean() / self.target_mean + 1e-6) **2 
        loss2 = torch.log(pcl2_object.mean() / self.target_mean + 1e-6) **2
        loss = loss1 + loss2

        return loss
    

# local spatial type consistency loss
class LocalSegmentConsistencyLoss(nn.Module):

    def __init__(self, params, kernel_size=3, loss_type='mse'):
        super(LocalSegmentConsistencyLoss, self).__init__()
        self.params = params
        self.kernel_size = kernel_size
        self.loss_type = loss_type.lower()
        assert self.loss_type in ['mse', 'kl'], "loss_type must be 'mse' or 'kl'"

    def forward(self, pcl):  

        pcl_soft = F.softmax(pcl, dim=1)
        # cal mean in every class
        avg_probs = F.avg_pool3d(
            pcl_soft, 
            kernel_size=self.kernel_size, 
            stride=1, 
            padding=self.kernel_size // 2
        )

        if self.loss_type == 'mse':
            loss = F.mse_loss(pcl_soft, avg_probs)
        elif self.loss_type == 'kl':
            # add eps to avoid log(0)
            eps = 1e-6
            loss = (pcl_soft * ((pcl_soft + eps) / (avg_probs + eps)).log()).sum(dim=1).mean()

        return loss


# local spatial type consistency loss
class LocalFlowSmoothLoss(nn.Module):

    def __init__(self, params, kernel_size=7, loss_type='mse', alpha = 0.8):
        super(LocalFlowSmoothLoss, self).__init__()
        self.params = params
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 == 1, "Kernel size must be odd."
        self.loss_type = loss_type.lower()
        assert self.loss_type in ['mse', 'kl'], "loss_type must be 'mse' or 'kl'"
        self.padding = kernel_size // 2

        # weight of Euro-distanse：shape = (1, 1, K, K, K)
        grid = torch.stack(torch.meshgrid(
            torch.arange(kernel_size),
            torch.arange(kernel_size),
            torch.arange(kernel_size),
            indexing='ij'
        ), dim=0).float()  # shape: (3, K, K, K)

        center = self.padding
        offsets = grid.permute(1, 2, 3, 0) - center  # (K, K, K, 3)
        dist = torch.norm(offsets, dim=-1)  # (K, K, K)

        self.weights = torch.softmax(torch.exp(-dist / alpha).reshape(-1), dim=0)\
            .reshape(1, 1, kernel_size, kernel_size, kernel_size).to('cuda')
        self.weights = self.weights.repeat(3, 1, 1, 1, 1)

    def forward(self, pcl1, flow, coords):  

        coords = coords.unsqueeze(0)
        # preprocess
        pcl1 = F.softmax(pcl1, dim=1)
        pcl1_object = pcl1[:,1,:,:,:]
        pcl1_object = pcl1_object.unsqueeze(1) # B * 1 * R * A * E
        # cal flow from index to real coords
        r_bin = self.params['radar_FFT_arr']['info_rae'][0][1]
        a_bin = self.params['radar_FFT_arr']['info_rae'][1][1]
        e_bin = self.params['radar_FFT_arr']['info_rae'][2][1]
        device = flow.device
        bin_tensor = torch.tensor([r_bin, a_bin, e_bin], dtype=flow.dtype, device=device).view(1, 3, 1, 1, 1)
        flow_metric = flow * bin_tensor  # shape remains: (B, 3, R, A, E)
        # cal flow loss
        flow1 = flow_metric * pcl1_object
        # cal smothless loss
        smoothed_flow  = F.avg_pool3d(flow1, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        dif_flow = flow1 - smoothed_flow
        dif_flow = dif_flow ** 2
        weight_dif_flow = F.conv3d(dif_flow, weight=self.weights, padding=self.padding, groups=3)
        weight_dif_flow = weight_dif_flow.sum(dim=1, keepdim=True)

        loss = weight_dif_flow.sum() / pcl1_object.sum()
    
        return loss