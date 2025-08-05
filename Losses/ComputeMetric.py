import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from Utils import IOStream

class ComputeMetric():
    def __init__(self, params):
        super(ComputeMetric, self).__init__()
        self.params = params
        self.snr_metric = []
        self.flow_metric = []
        self.chamfer_distance = []
        self.pdpfa_metric = []
        self.invalid = 0
        arrR, arrA, arrE = params['radar_FFT_arr']['arrRange'], \
                        params['radar_FFT_arr']['arrAzimuth'], params['radar_FFT_arr']['arrElevation']
        r_min, r_max, a_min, a_max, e_min, e_max = arrR.min(), arrR.max(), arrA.min(), arrA.max(), arrE.min(), arrE.max()
        # rae dimension
        r_dim, a_dim, e_dim = len(arrR), len(arrA), len(arrE)
        r_coords = torch.linspace(r_min, r_max, r_dim).view(r_dim, 1, 1).expand(r_dim, a_dim, e_dim)
        a_coords = torch.linspace(a_min, a_max, a_dim).view(1, a_dim, 1).expand(r_dim, a_dim, e_dim)
        e_coords = torch.linspace(e_min, e_max, e_dim).view(1, 1, e_dim).expand(r_dim, a_dim, e_dim)
        coords = torch.stack([r_coords, a_coords, e_coords], dim=0)
        coords = coords.to(dtype=torch.float)
        self.coords = coords
    
    def compute_pointflow_metric(self, x1, pcl, flow, pcls_label, flows_label):

        def compute_pd_pfa(seg, label_seg):
            label_seg = label_seg.squeeze(0)
            label_seg = label_seg > 0.9
            TP = (seg & label_seg).sum().item()
            FP = (seg & (~label_seg)).sum().item()
            FN = ((~seg) & label_seg).sum().item()
            TN = ((~seg) & (~label_seg)).sum().item()
            Pfa = FP / (FP + TN + 1e-6)  # sum a epsilon
            Pd = TP / (TP + FN + 1e-6)
            return Pfa, Pd

        def compute_detected_snr_stats_batch(echo_tensor: torch.Tensor,
                                     target_mask: torch.Tensor,
                                     ref_window_size: int = 5,
                                     guard_size: int = 0):
            signal = echo_tensor[target_mask]
            mask = ~target_mask
            noise = echo_tensor[mask]
            snr_db = 10 * (signal.mean() - noise.mean())
            return snr_db

        def chamfer_dist(pcd1, pcd2):
            # pcd1: (N1, 3), pcd2: (N2, 3)
            dist = torch.cdist(pcd1, pcd2, p=2)  

            # find nearest distance
            min1 = dist.min(dim=1)[0]  # (N1,)
            min2 = dist.min(dim=0)[0]  # (N2,)
            return min1.mean() + min2.mean()
        
        def spherical_to_cartesian(spherical_points):
            range_, azimuth, elevation = spherical_points[:, 0], spherical_points[:, 1], spherical_points[:, 2]
            # Calculate x, y, z from spherical coordinates
            x = range_ * torch.cos(elevation) * torch.cos(azimuth)
            y = range_ * torch.cos(elevation) * torch.sin(azimuth)
            z = range_ * torch.sin(elevation)
            # Stack the Cartesian coordinates into a tensor of shape (N, 3)
            return torch.stack([x, y, z], dim=1)
        
        def compute_epe(flow_est, flow_gt):
            # cal End-Point Error
            epe = torch.norm(flow_est - flow_gt, p=2, dim=-1)  
            return epe 

        def compute_rne(flow_est, flow_gt):
            # cal Relative Normalized Error
            norm_est = torch.norm(flow_est, p=2, dim=-1)
            norm_gt = torch.norm(flow_gt, p=2, dim=-1)
            rne = torch.norm(flow_est - flow_gt, p=2, dim=-1) / (norm_gt + 1e-6) 
            return rne

        def compute_accs(epe, rne, epe_threshold=0.05, rel_error_threshold=0.05):
            strict_condition = (epe < epe_threshold) | (rne < rel_error_threshold)
            acc_s = torch.sum(strict_condition).float() / strict_condition.numel()
            return acc_s

        def compute_accr(epe, rne, epe_threshold=0.1, rel_error_threshold=0.1):
            relaxed_condition = (epe < epe_threshold) | (rne < rel_error_threshold)
            acc_r = torch.sum(relaxed_condition).float() / relaxed_condition.numel()
            return acc_r
        
        # preprocess
        x1 = torch.max(x1, dim=1, keepdim=True)
        x1 = x1.values
        B, _, R, A, E = pcls_label.shape
        pcl = pcl.permute(0,2,1).contiguous().view(B, -1, R, A, E)
        flow = flow.permute(0,2,1).contiguous().view(B, -1, R, A, E)
        r_bin, a_bin, e_bin = self.params['radar_FFT_arr']['info_rae'][0][1], self.params['radar_FFT_arr']['info_rae'][1][1], self.params['radar_FFT_arr']['info_rae'][2][1]
        device = flow.device
        coords = self.coords.to(device)
        bin_tensor = torch.tensor([r_bin, a_bin, e_bin], dtype=flow.dtype, device=device).view(1, 3, 1, 1, 1)
        flow = flow * bin_tensor  # shape remains: (B, 3, R, A, E)
        flows_label = flows_label * bin_tensor
        # take point
        pcl_sigmoid = pcl.sigmoid()
        mask_pred = (pcl_sigmoid > 0.5).float()
        mask_pred = mask_pred.bool()
        # snr 
        snr = compute_detected_snr_stats_batch(x1, mask_pred)
        # chamfer
        pred_pcl_results = []
        for b in range(B):
            m = mask_pred[b, 0] # (R, A, E)
            c = coords # (3, R, A, E)
            selected = c[:, m].T  # (3, N) to (N, 3)
            selected = spherical_to_cartesian(selected)
            pred_pcl_results.append(selected)
        # take gt point
        mask_gt = (pcls_label > 0.9).float()
        mask_gt = mask_gt.bool()
        self.snr_metric.append((snr.item()))
        pd, pfa = compute_pd_pfa(mask_pred, mask_gt)
        self.pdpfa_metric.append((pd.item(), pfa.item()))
        gt_pcl_results = []
        for b in range(B):
            m = mask_gt[b, 0] # (R, A, E)
            c = coords # (3, R, A, E)
            selected = c[:, m].T  # (3, N) to (N, 3)
            selected = spherical_to_cartesian(selected)
            gt_pcl_results.append(selected)
        # cal chamfer dist
        chamfer_dists = []
        for pred, gt in zip(pred_pcl_results, gt_pcl_results):
            if pred.shape[0] == 0:
                self.invalid += 1   # invalid pcl
                print(f'invalid pcl result, all invalid num = {self.invalid}')
                continue
            print(f'pred-gt shape: {pred.shape[0]}-{gt.shape[0]}')
            dist = chamfer_dist(pred, gt)
            chamfer_dists.append(dist)
        chamfer_dists = torch.stack(chamfer_dists)
        avg_chamfer = chamfer_dists.mean()
        self.chamfer_distance.extend(chamfer_dists)
        # take flow
        flow_metric = []
        for b in range(B):
            m = mask_gt[b, 0] # (R, A, E)
            c = coords # (3, R, A, E)
            pf = flow[b] 
            gtf = flows_label[b]
            c = c[:, m].T # (3, N) to (N, 3)
            pf = pf[:, m].T  
            gtf = gtf[:, m].T
            # Please note the flow is mean (dr, da, de) in Spherical Coordinate
            valid_flow_warp = c + pf
            valid_coords_in_Cartesian = spherical_to_cartesian(c)
            valid_flow_warp_in_Cartesian = spherical_to_cartesian(valid_flow_warp)
            valid_flow_cartesian = valid_flow_warp_in_Cartesian - valid_coords_in_Cartesian
            # Please note the flowlabel is mean (dr, da, de) in Spherical Coordinate
            valid_flow_warpgt = c + gtf
            valid_coords_in_Cartesiangt = spherical_to_cartesian(c)
            valid_flow_warp_in_Cartesiangt = spherical_to_cartesian(valid_flow_warpgt)
            valid_flow_cartesiangt = valid_flow_warp_in_Cartesiangt - valid_coords_in_Cartesiangt
            # cal metric
            epe = compute_epe(valid_flow_cartesian, valid_flow_cartesiangt)
            rne = compute_rne(valid_flow_cartesian, valid_flow_cartesiangt)
            accs = compute_accs(epe, rne)
            accr = compute_accr(epe, rne)
            flow_metric.append((epe.mean().item(), rne.mean().item(), accs.mean().item(), accr.mean().item()))
        self.flow_metric.extend(flow_metric)
        epes = torch.tensor([m[0] for m in flow_metric])
        rnes = torch.tensor([m[1] for m in flow_metric])
        accs_list = torch.tensor([m[2] for m in flow_metric])
        accr_list = torch.tensor([m[3] for m in flow_metric])
        avg_epe = epes.mean()
        avg_rne = rnes.mean()
        avg_accs = accs_list.mean()
        avg_accr = accr_list.mean()
        return snr.item(), avg_chamfer.item(), pd.item(), pfa.item(),\
                    avg_epe.item(), avg_rne.item(), avg_accs.item(), avg_accr.item()
        
    def output_metric(self):
        # pd pfa
        pd_mean = torch.tensor([m[0] for m in self.pdpfa_metric])
        avg_pd_mean = pd_mean.mean()
        pfa_mean = torch.tensor([m[1] for m in self.pdpfa_metric])
        avg_pfa_mean = pfa_mean.mean()
        # snr
        snr_mean = torch.tensor([m[0] for m in self.snr_metric])
        avg_snr_mean = snr_mean.mean()
        # chamfer
        chamfer_dists = torch.stack(self.chamfer_distance)
        avg_chamfer = chamfer_dists.mean()
        # flow
        epes = torch.tensor([m[0] for m in self.flow_metric])
        rnes = torch.tensor([m[1] for m in self.flow_metric])
        accs_list = torch.tensor([m[2] for m in self.flow_metric])
        accr_list = torch.tensor([m[3] for m in self.flow_metric])
        avg_epe = epes.mean()
        avg_rne = rnes.mean()
        avg_accs = accs_list.mean()
        avg_accr = accr_list.mean()
        return avg_snr_mean.item(), avg_chamfer.item(), avg_pd_mean.item(), avg_pfa_mean.item(),\
            avg_epe.item(), avg_rne.item(), avg_accs.item(), avg_accr.item()
    
    def compute_flow_in_cartesian(self, pcl, flow, pcls_label, flows_label):

        def spherical_to_cartesian(spherical_points):
            range_, azimuth, elevation = spherical_points[:, 0], spherical_points[:, 1], spherical_points[:, 2]
            # Calculate x, y, z from spherical coordinates
            x = range_ * torch.cos(elevation) * torch.cos(azimuth)
            y = range_ * torch.cos(elevation) * torch.sin(azimuth)
            z = range_ * torch.sin(elevation)
            # Stack the Cartesian coordinates into a tensor of shape (N, 3)
            return torch.stack([x, y, z], dim=1)
        B, _, R, A, E = pcls_label.shape
        pcl = pcl.permute(0,2,1).contiguous().view(B, -1, R, A, E)
        flow = flow.permute(0,2,1).contiguous().view(B, -1, R, A, E)
        r_bin, a_bin, e_bin = self.params['radar_FFT_arr']['info_rae'][0][1], self.params['radar_FFT_arr']['info_rae'][1][1], self.params['radar_FFT_arr']['info_rae'][2][1]
        device = flow.device
        coords = self.coords.to(device)
        bin_tensor = torch.tensor([r_bin, a_bin, e_bin], dtype=flow.dtype, device=device).view(1, 3, 1, 1, 1)
        flow = flow * bin_tensor  # shape remains: (B, 3, R, A, E)
        flows_label = flows_label * bin_tensor
        # take point
        pcl_sigmoid = pcl.sigmoid()
        mask_pred = (pcl_sigmoid > 0.9).float()
        mask_pred = mask_pred.bool()
        # chamfer
        mpd = mask_pred[0, 0] # (R, A, E)
        pcl_coord = coords[:, mpd].T  # (3, N) to (N, 3)
        pcl_coord = spherical_to_cartesian(pcl_coord)
        # take gt point
        mask_gt = (pcls_label > 0.9).float()
        mask_gt = mask_gt.bool()
        mgt = mask_gt[0, 0] # (R, A, E)
        gtpcl_coord = coords[:, mgt].T  # (3, N) to (N, 3)
        gtpcl_coord = spherical_to_cartesian(gtpcl_coord)
        # flow 
        pf = flow[0] 
        gtf = flows_label[0]
        pf = pf[:, mpd].T  
        gtf = gtf[:, mgt].T
        # Please note the flow is mean (dr, da, de) in Spherical Coordinate
        c = coords[:, mpd].T
        valid_flow_warp = c + pf
        valid_coords_in_Cartesian = spherical_to_cartesian(c)
        valid_flow_warp_in_Cartesian = spherical_to_cartesian(valid_flow_warp)
        valid_flow_cartesian = valid_flow_warp_in_Cartesian - valid_coords_in_Cartesian
        # Please note the flowlabel is mean (dr, da, de) in Spherical Coordinate
        c = coords[:, mgt].T
        valid_flow_warpgt = c + gtf
        valid_coords_in_Cartesiangt = spherical_to_cartesian(c)
        valid_flow_warp_in_Cartesiangt = spherical_to_cartesian(valid_flow_warpgt)
        valid_flow_cartesiangt = valid_flow_warp_in_Cartesiangt - valid_coords_in_Cartesiangt
        return pcl_coord.cpu().numpy(), valid_flow_cartesian.cpu().numpy(), gtpcl_coord.cpu().numpy(), valid_flow_cartesiangt.cpu().numpy()
    
    def compute_flow_in_lidar(self, dict_cfg):

        def load_lidar(path_pcd, len_header=11, ego_offset=1e-3):
            f = open(path_pcd, 'r')
            lines = f.readlines()
            f.close()
            list_header = lines[:len_header]
            list_values = lines[len_header:]
            list_values = list(map(lambda x: x.split(' '), list_values))
            values = np.array(list_values, dtype=np.float32)
            values = values[ # delete (0,0)
            np.where(
                (values[:,0]<-ego_offset) | (values[:,0]>ego_offset) |  # x
                (values[:,1]<-ego_offset) | (values[:,1]>ego_offset)    # y
            )]
            return values
        
        def calib_lidar(values, list_calib_xyz):
            arr_calib_xyz = np.array(list_calib_xyz, dtype=values.dtype).reshape(1,3)
            arr_calib_xyz = arr_calib_xyz.repeat(repeats=len(values), axis=0)
            values[:,:3] += arr_calib_xyz
            return values
        
        def cartesian_to_spherical(x, y, z):
            range_ = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            azimuth = np.arctan2(y, x)  
            elevation = np.arcsin(np.clip(z / np.maximum(range_, 1e-8), -1.0, 1.0))
            return np.stack([range_, azimuth, elevation], axis=1)
        
        def spherical_to_cartesian(spherical_points):
            range_, azimuth, elevation = spherical_points[:, 0], spherical_points[:, 1], spherical_points[:, 2]
            # Calculate x, y, z from spherical coordinates
            x = range_ * np.cos(elevation) * np.cos(azimuth)
            y = range_ * np.cos(elevation) * np.sin(azimuth)
            z = range_ * np.sin(elevation)

            # Stack into shape (N, 3)
            return np.stack([x, y, z], axis=1)
        
        def roi_filter(point, x_axis, y_axis, z_axis):
            num_x = len(x_axis)
            num_y = len(y_axis)
            num_z = len(z_axis)
            # Calculate voxel indices for each axis using broadcasting
            x_indices = np.searchsorted(x_axis, point[..., 0], side='left')
            y_indices = np.searchsorted(y_axis, point[..., 1], side='left')
            z_indices = np.searchsorted(z_axis, point[..., 2], side='left')

            # This is just in case there are some points outside the grid. It should not be the case since we clean first
            # the lidar point cloud.
            valid_indices = (x_indices > 0) & (x_indices < num_x) & (y_indices > 0) & (y_indices < num_y) & (
                    z_indices > 0) & (z_indices < num_z)
            point_roi = point[valid_indices, :]
            return point_roi
        
        def transform_and_convert(coords, mask, t_ego_bbx, flows):
            xyz = coords[mask]  # shape (N_sel, 3)
            ones = np.ones((xyz.shape[0], 1), dtype=xyz.dtype)
            xyz1 = np.hstack([xyz, ones])  # (N_sel, 4)
            transformed = (t_ego_bbx @ xyz1.T).T[:, :3] 
            flows[mask] = transformed - xyz
            return flows
        
        lidar_path = dict_cfg['lidar'][0]
        label_path = dict_cfg['label'][0]
        calib = dict_cfg['calib']
        calib_list = [x.item() for x in calib]

        lidar_point = load_lidar(lidar_path)
        lidar_point = calib_lidar(lidar_point, calib_list[-3:])
        # roi filter
        lidar_point_polar = cartesian_to_spherical(lidar_point[:, 0], lidar_point[:, 1], lidar_point[:, 2])
        lidar_point_polar = roi_filter(lidar_point_polar, self.params['radar_FFT_arr']['arrRange'], \
                        self.params['radar_FFT_arr']['arrAzimuth'], self.params['radar_FFT_arr']['arrElevation'])
        lidar_point = spherical_to_cartesian(lidar_point_polar)
        # flow
        label = np.load(label_path, allow_pickle=True).item()
        N = lidar_point.shape[0]
        flows = np.zeros_like(lidar_point)  # (N, 3)
        mask_bg = np.full((N,), True, dtype=bool)
        t_rigid = np.array(label['gt_rigid_trans'], dtype=np.float32)  # (4, 4)
        flows = transform_and_convert(lidar_point, mask_bg, t_rigid, flows)
        # fg trans
        for fg_trans in label['gt_fg_rigid_trans']:
            bbox_in_rae, t_ego_bbx, _,_,_,_,_ = fg_trans
            r_minbbox, r_maxbbox, a_minbbox, a_maxbbox, e_minbbox, e_maxbbox = bbox_in_rae
            mask_fg = (lidar_point_polar[:,0] >= r_minbbox) & (lidar_point_polar[:,0] <= r_maxbbox) & \
                        (lidar_point_polar[:,1] >= a_minbbox) & (lidar_point_polar[:,1] <= a_maxbbox) & \
                        (lidar_point_polar[:,2] >= e_minbbox) & (lidar_point_polar[:,2] <= e_maxbbox)
            t_ego_bbx = np.array(t_ego_bbx, dtype=np.float32)
            flows = transform_and_convert(lidar_point, mask_fg, t_ego_bbx, flows)

        return lidar_point, flows
        
        