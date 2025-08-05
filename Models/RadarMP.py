import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.DopplerMLP import DopplerMLP
from Models.ResNet3D import ResNet3D, DenseConv3d
from Models.Motion3DPerception import Motion3DPerception
from Models.Motion2DPerception import Motion2DPerception
from Models.Head import SegHead, FlowHead, SegFlowHead
from Models.SegAttention import SegAttention

class RadarMP(nn.Module):
    def __init__(self, params):
        super(RadarMP, self).__init__()
        self.params = params
        doppler_dim = self.params['radar_tensor_dim']['doppler']
        doppler_encoderdim = self.params['radarmp']['dopmlp']['output_channel']
        backbone_baseplanes = self.params['radarmp']['resnet3d']['baseplane']
        backbone_widenfactor = self.params['radarmp']['resnet3d']['widen_factor']
        # seg
        seg_rae64_att = self.params['radarmp']['seg']['seg_rae64_att']
        seg_ae_att = self.params['radarmp']['seg']['seg_ae_att']
        seg_outdim = self.params['radarmp']['seg']['output_dims']
        seg_head = self.params['radarmp']['seg']['num_heads']
        seg_layer = self.params['radarmp']['seg']['num_layers']
        # corr
        correlation_head  = self.params['radarmp']['correlation']['num_heads']
        correlation_point = self.params['radarmp']['correlation']['num_points']
        correlation_level = self.params['radarmp']['correlation']['num_levels']
        correlation_dims  = self.params['radarmp']['correlation']['embed_dims']
        correlation_out   = self.params['radarmp']['correlation']['output_dims']
        # initial encoder
        self.dopmlp = DopplerMLP(input_channels=doppler_dim, coord_channels=doppler_dim, output_channels=doppler_encoderdim, dropout=0.0)
        self.backbone3d = ResNet3D(baseplanes=backbone_baseplanes, input_channels=doppler_encoderdim, widen_factor=backbone_widenfactor)
        # correlation
        value_dims_list = [backbone_baseplanes, backbone_baseplanes*2, backbone_baseplanes*4]
        self.motion3d = Motion3DPerception(value_dims_list=value_dims_list, num_levels=correlation_level, num_points=correlation_point,
                    num_heads=correlation_head, embed_dims=correlation_dims, output_dims=correlation_out)
        # freeze motion2d
        self.motion2d = Motion2DPerception()
        flow2dmodule_path = self.params.get('path', {}).get('flow2dmodel_path')
        if flow2dmodule_path is not None:
            if os.path.exists(flow2dmodule_path):
                self.motion2d.load_state_dict(torch.load(flow2dmodule_path))
            # if motion2d need freeze
            # for p in self.motion2d.parameters():
                # p.requires_grad = False
        # self seg
        # self.seghead = SegHead()
        self.segattention = SegAttention(RAE64_channels=correlation_out*64, AE_channels=correlation_out*seg_ae_att, output_channels=seg_outdim, 
                                         rae64_d_model=seg_rae64_att, ae_d_model=seg_ae_att, num_heads=seg_head, num_layers=seg_layer)
        self.head = SegFlowHead(seg_channels=seg_outdim, corr_channels=correlation_out)
        # self.head = SegFlowHead(seg_channels=correlation_out, corr_channels=correlation_out)

    def forward(self, x1, x2, dop_arr, coords, ori):
        # preprocess
        device = x1.device
        bs, dop, ran, azi, ele = x1.shape
        coords_RAE64 = coords.view(bs, 3, ran//4, 4, azi//4, 4, ele//4, 4).permute(0, 2, 4, 6, 3, 5, 7, 1)
        coords_RAE64 = coords_RAE64.contiguous().view(bs, (ran//4*azi//4*ele//4), 3*4*4*4)
        ori_AE = ori.mean(dim=2)  # (B, 3, azi, ele)
        ori_AE = ori_AE.permute(0, 2, 3, 1).contiguous().view(bs, azi*ele, 3)
        # seginit = self.seghead(x1)
        # flow2d
        flow_cube_layers = []
        x1_ra = x1.mean(dim=(1, 4));x1_re = x1.mean(dim=(1, 3)); x1_ae = x1.mean(dim=(1, 2))
        x2_ra = x2.mean(dim=(1, 4));x2_re = x2.mean(dim=(1, 3)); x2_ae = x2.mean(dim=(1, 2))
        x1_ra = x1_ra.unsqueeze(1);x1_re = x1_re.unsqueeze(1);x1_ae = x1_ae.unsqueeze(1)
        x2_ra = x2_ra.unsqueeze(1);x2_re = x2_re.unsqueeze(1);x2_ae = x2_ae.unsqueeze(1)
        flow_ra, flow_re, flow_ae = self.motion2d(x1_ra, x2_ra, x1_re, x2_re, x1_ae, x2_ae)
        for flow_ra_layer, flow_re_layer, flow_ae_layer in zip(flow_ra, flow_re, flow_ae):
            _, _, ranl, azil = flow_ra_layer.shape
            _, _, ranl, elel = flow_re_layer.shape
            flow_ra_e_layer = flow_ra_layer.unsqueeze(4).expand(-1, -1, -1, -1, elel)
            flow_re_a_layer = flow_re_layer.unsqueeze(3).expand(-1, -1, -1, azil, -1)
            flow_ae_r_layer = flow_ae_layer.unsqueeze(2).expand(-1, -1, ranl, -1, -1)
            flow_r_layer = (flow_ra_e_layer[:, 0] + flow_re_a_layer[:, 0]) / 2
            flow_a_layer = (flow_ra_e_layer[:, 1] + flow_ae_r_layer[:, 0]) / 2
            flow_e_layer = (flow_re_a_layer[:, 1] + flow_ae_r_layer[:, 1]) / 2 
            flow_cube_layer = torch.stack([flow_r_layer, flow_a_layer, flow_e_layer], dim=1) # # bs 3 ranl azil elel
            flow_cube_layers.append(flow_cube_layer)
        # encode x1
        x1_seq = x1.permute(0,2,3,4,1).contiguous().view(bs, ran*azi*ele, dop)
        x1_seq = self.dopmlp(x1_seq, dop_arr)
        x1_seq = x1_seq.permute(0,2,1).contiguous().view(bs, -1, ran, azi, ele)
        x1_msf = self.backbone3d(x1_seq)
        # encode x2
        x2_seq = x2.permute(0,2,3,4,1).contiguous().view(bs, ran*azi*ele, dop)
        x2_seq = self.dopmlp(x2_seq, dop_arr)
        x2_seq = x2_seq.permute(0,2,1).contiguous().view(bs, -1, ran, azi, ele)
        x2_msf = self.backbone3d(x2_seq)
        # corr
        x1x2, spatial_shapes = self.motion3d(x1_msf, x2_msf, flow_cube_layers)
        x1x2_msf = []
        start_idx = 0
        for shape in spatial_shapes:
            ranl, azil, elel = shape
            num_token = ranl*azil*elel  # token num
            x1x2_l = x1x2[:, start_idx:start_idx+num_token, :]  # [B, N, C]
            x1x2_l = x1x2_l.permute(0,2,1).contiguous().view(bs, -1, ranl, azil, elel)  
            x1x2_l = F.interpolate(x1x2_l, size=(ran, azi, ele), mode='trilinear', align_corners=False)
            x1x2_msf.append(x1x2_l)
            start_idx += num_token
        # seg
        x1x2_fpn = sum(x1x2_msf)
        x1_seg = self.segattention(x1x2_fpn, coords_RAE64, ori_AE)
        # output
        x1x2_fpn = x1x2_fpn.permute(0,2,3,4,1).contiguous().view(bs, ran*azi*ele, -1)
        seg, flow = self.head(x1_seg, x1x2_fpn)
        
        
        return seg, flow, x1x2_fpn
    


    

