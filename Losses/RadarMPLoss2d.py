import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import numpy as np

class PowerFlow2DLoss(nn.Module):
    def __init__(self, params):
        super(PowerFlow2DLoss, self).__init__()
        self.params = params
        self.loss_fn = nn.L1Loss()
    
    def warp_2d(self, x, flo):
        B, C, H, W = x.shape
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()
        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo
        # scale grid to [-1,1] 
        vgrid_norm = torch.zeros_like(vgrid)
        vgrid_norm[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / (W - 1) - 1.0
        vgrid_norm[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / (H - 1) - 1.0
        vgrid_norm = vgrid_norm.permute(0, 2, 3, 1)  # [B, H, W, 2]
        output = F.grid_sample(x, vgrid_norm, mode='bilinear', padding_mode='border', align_corners=True)
        return output


    def forward(self, flow_ra, flow_re, flow_ae, x1_ra, x2_ra, x1_re, x2_re, x1_ae, x2_ae): 
        # x1 and x2 to 2D
        loss_ra_sum = 0.0
        loss_re_sum = 0.0
        loss_ae_sum = 0.0
        # recurrent
        for i, (flow_ra_layer, flow_re_layer, flow_ae_layer) in enumerate(zip(flow_ra, flow_re, flow_ae)):
            x1_ra_layer = F.interpolate(x1_ra, scale_factor=0.5**i, mode='bilinear', align_corners=False, antialias=True)
            x2_ra_layer = F.interpolate(x2_ra, scale_factor=0.5**i, mode='bilinear', align_corners=False, antialias=True)
            x1_re_layer = F.interpolate(x1_re, scale_factor=0.5**i, mode='bilinear', align_corners=False, antialias=True)
            x2_re_layer = F.interpolate(x2_re, scale_factor=0.5**i, mode='bilinear', align_corners=False, antialias=True)
            x1_ae_layer = F.interpolate(x1_ae, scale_factor=0.5**i, mode='bilinear', align_corners=False, antialias=True)
            x2_ae_layer = F.interpolate(x2_ae, scale_factor=0.5**i, mode='bilinear', align_corners=False, antialias=True)
            assert x2_ra_layer.shape[-2:] == flow_ra_layer.shape[-2:]
            # warp
            x2_ra_warped = self.warp_2d(x2_ra_layer, flow_ra_layer)
            x2_re_warped = self.warp_2d(x2_re_layer, flow_re_layer)
            x2_ae_warped = self.warp_2d(x2_ae_layer, flow_ae_layer)
            # cal loss
            loss_ra = self.loss_fn(x1_ra_layer, x2_ra_warped)
            loss_re = self.loss_fn(x1_re_layer, x2_re_warped)
            loss_ae = self.loss_fn(x1_ae_layer, x2_ae_warped)
            loss_ra_sum = loss_ra_sum + loss_ra * 0.8**i
            loss_re_sum = loss_re_sum + loss_re * 0.8**i
            loss_ae_sum = loss_ae_sum + loss_ae * 0.8**i

        loss = 0.6 * loss_ra_sum + 0.35 * loss_re_sum + 0.05 * loss_ae_sum

        loss_item = {
            'total_loss': loss, 
            'ra_loss': loss_ra_sum,
            're_loss': loss_re_sum,
            'ae_loss': loss_ae_sum,
        }

        return loss, loss_item