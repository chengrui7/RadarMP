import scipy.io
import sys
import numpy as np
import open3d as o3d
import torch 
import torch.nn.functional as F
import itertools
import math

from scipy.spatial import cKDTree


def chamfer_dist(pcl0, pcl1):
    """
    Load two point clouds, project the first onto the second using the given transform,
    and compute the Chamfer distance between the transformed and second point clouds.
    
    Returns:
    - Chamfer distance: float
    """
    # Compute Chamfer distance
    # Create KDTree for fast nearest-neighbor search
    tree0 = cKDTree(pcl0)
    tree1 = cKDTree(pcl1)
    
    # Distance from pcl1 to pcl2
    dist0, _ = tree0.query(pcl1, k=1)
    # Distance from pcl2 to pcl1
    dist1, _ = tree1.query(pcl0, k=1)
    
    # Chamfer distance
    chamfer_dist = np.mean(dist0**2) + np.mean(dist1**2)
    return chamfer_dist

def get_rigid_flow(pcl, ego_trans):

    #  pcl to homogeneous
    pcl_homogeneous = np.hstack((pcl, np.ones((pcl.shape[0], 1))))
    pcl_tran_homogeneous = np.dot(pcl_homogeneous, ego_trans.T)
    pcl_tran = pcl_tran_homogeneous[:, :3]
    # flow_r
    flow_r = pcl_tran - pcl
    
    return flow_r

def get_fg_rigid_flow(label1, label2, pcl, pcl_t, info_rae):

    def get_rigid_flow(pcl, ego_trans):
        pcl_homogeneous = np.hstack((pcl, np.ones((pcl.shape[0], 1))))
        pcl_tran_homogeneous = np.dot(pcl_homogeneous, ego_trans.T)
        pcl_tran = pcl_tran_homogeneous[:, :3]
        flow_r = pcl_tran - pcl
        return flow_r

    def inv_se3(T):
        Tinv = np.eye(4)
        Tinv[:3, :3] = T[:3, :3].T  # Transpose of the rotation matrix
        Tinv[:3, 3] = -np.dot(Tinv[:3, :3], T[:3, 3])  # -R^T * t
        return Tinv

    def get_inbox_flow(pnts, t_ego_bbx1, t_ego_bbx2):
        t_bbx1_bbx2 = np.dot(t_ego_bbx2, inv_se3(t_ego_bbx1))
        print(t_bbx1_bbx2)
        inbox_flow = get_rigid_flow(pnts, t_bbx1_bbx2)
        return inbox_flow, t_bbx1_bbx2

    def get_bbx_transformation(bbx):
        t_ego_bbx = np.eye(4)
        t_ego_bbx[:3,:3] = bbx.R
        t_ego_bbx[:3,3] = bbx.center
        return t_ego_bbx
    
    def kabsch_algorithm_with_matching(P, Q):
        tree_P = cKDTree(P)
        distances, indices = tree_P.query(Q, k=1) 
        P_matched = P[indices]
        centroid_P = np.mean(P_matched, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        P_centered = P_matched - centroid_P
        Q_centered = Q - centroid_Q

        H = np.dot(P_centered.T, Q_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = centroid_Q - np.dot(R, centroid_P)

        t_ego_bbx = np.eye(4)
        t_ego_bbx[:3,:3] = R
        t_ego_bbx[:3,3] = t
        print(t_ego_bbx)
        return t_ego_bbx
    
    def is_bbox_in_rae(corners_rotated, info_rae):
        # 
        r_min, r_bin, r_max = info_rae[0]
        a_min, a_bin, a_max = info_rae[1]
        e_min, e_bin, e_max = info_rae[2]

        # 
        x, y, z = corners_rotated[:, 0], corners_rotated[:, 1], corners_rotated[:, 2]
        R = np.sqrt(x**2 + y**2 + z**2)
        A = np.arctan2(y, x)
        E = np.arcsin(z / R)

        # 
        in_r = (r_min <= R) & (R <= r_max)
        in_a = (a_min <= A) & (A <= a_max)
        in_e = (e_min <= E) & (E <= e_max)

        r_minbbox, r_maxbbox = max(R.min(),r_min), min(R.max(),r_max)
        a_minbbox, a_maxbbox = max(A.min(),a_min), min(A.max(),a_max)
        e_minbbox, e_maxbbox = max(E.min(),e_min), min(E.max(),e_max)

        return np.any(in_r & in_a & in_e), [r_minbbox, r_maxbbox, a_minbbox, a_maxbbox, e_minbbox, e_maxbbox]

    # init
    num_pnts = pcl.shape[0]
    fg_idx = []
    fg_flow = np.zeros((num_pnts,3),dtype=np.float32)
    fg_trans = []

    # find track
    for uqi_obj in label1.tracking:
        cls_name, bbox_calib, idx_obj,idx_now,unique_id,_,_,velocity,corners_rotated,_,_,_ = uqi_obj
        bool_bbox_in_rae, bbox_in_rae = is_bbox_in_rae(corners_rotated, info_rae)
        if bool_bbox_in_rae:
            for track_obj in label2.tracking:
                cls_name_t, bbox_calib_t, _, _,unique_id_t,_,_,velocity_t,corners_rotated_t,_,_,_ = track_obj
                if unique_id == unique_id_t:
                
                    obbx1 = o3d.geometry.OrientedBoundingBox.create_from_points(
                            o3d.utility.Vector3dVector(corners_rotated))
                    obbx2 = o3d.geometry.OrientedBoundingBox.create_from_points(
                            o3d.utility.Vector3dVector(corners_rotated_t))
                    
                    pc1 = o3d.utility.Vector3dVector(pcl.astype(np.float64))
                    in_box_idx = obbx1.get_point_indices_within_bounding_box(pc1)
                    pc2 = o3d.utility.Vector3dVector(pcl_t.astype(np.float64))
                    in_box_idx_t = obbx2.get_point_indices_within_bounding_box(pc2)
                    
                    if len(in_box_idx)>0:
                        in_box_points = pcl[in_box_idx, 0:3]
                        in_box_points_t = pcl_t[in_box_idx_t, 0:3]
                        t_ego_bbx1 = get_bbx_transformation(obbx1)
                        t_ego_bbx2 = get_bbx_transformation(obbx2)
                        in_box_flow_bboxt, t_ego_bbox_label = get_inbox_flow(in_box_points, t_ego_bbx1, t_ego_bbx2)
                        t_ego_bbx_kabsch = kabsch_algorithm_with_matching(in_box_points, in_box_points_t)
                        in_box_flow_kabsch = get_rigid_flow(in_box_points, t_ego_bbx_kabsch)
                        if chamfer_dist(in_box_points+in_box_flow_bboxt, in_box_points_t) > chamfer_dist(in_box_points+in_box_flow_kabsch, in_box_points_t):
                            in_box_flow = in_box_flow_kabsch
                            t_ego_bbx = t_ego_bbx_kabsch
                        else:
                            in_box_flow = in_box_flow_bboxt
                            t_ego_bbx = t_ego_bbox_label
                        # avoid wrong labels caused by inaccurate MOT output                    
                        if np.linalg.norm(in_box_flow,axis=1).max()<3:
                            fg_flow[in_box_idx] = in_box_flow
                            fg_idx.extend(in_box_idx)
                            fg_trans.append([bbox_in_rae, t_ego_bbx, idx_obj, idx_now, unique_id, cls_name, bbox_calib])

    return fg_flow, fg_idx, fg_trans
                

def get_pseudolabel_pcl(tesseract, next_tesseract, coords, flows, device):
    # init
    tesseract = (tesseract - np.min(tesseract)) / (np.max(tesseract) - np.min(tesseract) + 1e-6)
    tesseract = torch.tensor(tesseract).to(dtype=torch.float)
    next_tesseract = (next_tesseract - np.min(next_tesseract)) / (np.max(next_tesseract) - np.min(next_tesseract) + 1e-6)
    next_tesseract = torch.tensor(next_tesseract).to(dtype=torch.float)
    # remove doppler dimension
    rae_tesseract = torch.max(tesseract, dim=0, keepdim=True).values
    rae_next_tesseract = torch.max(next_tesseract, dim=0, keepdim=True).values
    # coords + flow
    r_vals = coords[0]
    a_vals = coords[1]
    e_vals = coords[2]
    x_vals = r_vals * torch.cos(e_vals) * torch.cos(a_vals)
    y_vals = r_vals * torch.cos(e_vals) * torch.sin(a_vals)
    z_vals = r_vals * torch.sin(e_vals)
    xyzcoords = torch.stack([x_vals, y_vals, z_vals], dim=0)
    newxyzcoords = xyzcoords + flows
    # torch nearest beighbor search
    xyzcoords = xyzcoords.to(device)
    newxyzcoords = newxyzcoords.to(device)
    R, A, E = xyzcoords.shape[1], xyzcoords.shape[2], xyzcoords.shape[3]
    print(f'R: {R}, A: {A}, E: {E}')
    # Generate 3x3x3 neighborhood offsets (125 combinations)
    offsets = torch.tensor(list(itertools.product(range(-2, 3), repeat=3)), 
                        device=device)  # (125, 3)
    # Create original grid base index（H, W, D）
    r, a, e = torch.meshgrid(torch.arange(R, device=device),
                            torch.arange(A, device=device),
                            torch.arange(E, device=device),
                            indexing='ij')
    # Generate candidate index（H, W, D, 81）
    candidate_r = (r.unsqueeze(-1) + offsets[:, 0]).clamp(0, R-1)
    candidate_a = (a.unsqueeze(-1) + offsets[:, 1]).clamp(0, A-1)
    candidate_e = (e.unsqueeze(-1) + offsets[:, 2]).clamp(0, E-1)
    # Flatten candidate coordinates into linear indices（H*W*D*125,）
    linear_indices = (
        candidate_r * (A * E) + 
        candidate_a * E + 
        candidate_e
    ).view(-1)
    # Collect candidate coordinate values（3, H*W*D*125）
    xyzcoords_flat = xyzcoords.view(3, -1)
    candidate_xyzcoords = torch.gather(
        xyzcoords_flat, 
        1, 
        linear_indices.unsqueeze(0).expand(3, -1)
    ) 
    # Reshape to (3, H, W, D, 81) and calculate the distance
    candidate_xyzcoords = candidate_xyzcoords.view(3, R, A, E, 125)
    diffs = candidate_xyzcoords - newxyzcoords.unsqueeze(-1)  # (3, H, W, D, 125)
    dists = torch.sum(diffs ** 2, dim=0)               # (H, W, D, 125)
    # Find the minimum distance index
    min_indices = torch.argmin(dists, dim=-1)          # (H, W, D)
    # Get the corresponding offset and calculate the final index
    selected_offsets = offsets[min_indices.flatten()].view(R, A, E, 3)
    nearest_r = (r + selected_offsets[..., 0]).clamp(0, R-1)
    nearest_a = (a + selected_offsets[..., 1]).clamp(0, A-1)
    nearest_e = (e + selected_offsets[..., 2]).clamp(0, E-1)
    nearest_indices = torch.stack([nearest_r, nearest_a, nearest_e], dim=0)
    # cal pw dif at corrspond position
    raedevice = rae_tesseract.device
    r_prime = nearest_indices[0].long().to(raedevice)  # (256, 107, 37)
    a_prime = nearest_indices[1].long().to(raedevice)   # (256, 107, 37)
    e_prime = nearest_indices[2].long().to(raedevice)   # (256, 107, 37)
    # rae_tesseract[:, r, a, e] - rae_next_tesseract[:, r', a', e']
    rae_tesseract = rae_tesseract.squeeze(0)
    rae_next_tesseract = rae_next_tesseract.squeeze(0)
    intensity_diff = rae_tesseract - rae_next_tesseract[r_prime, a_prime, e_prime] 
    percentage_diff = torch.abs(intensity_diff) / torch.abs(rae_tesseract)
    percentage_diff = percentage_diff.unsqueeze(0)
    threshold = 0.001
    pcl_label = percentage_diff < threshold
    indices = pcl_label.nonzero()
    indices_list = indices.tolist()
    return indices_list
        
# Load Radar Flow Cube
def label2cube(radar_cube, label):
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
    # init
    arrR, arrA, arrE = radar_cube.arr_range, radar_cube.arr_azimuth, radar_cube.arr_elevation
    r_min, r_max, a_min, a_max, e_min, e_max = arrR.min(), arrR.max(), arrA.min(), arrA.max(), arrE.min(), arrE.max()

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
    return coords, flows


def lidar2labelpcl(lidar_pc, arrR, arrA, arrE):
    """
    It converts the lidar point cloud into a lidar cube using the same axis as the radar cube.
    The axis are not uniform.

    :param lidar_pc: the lidar point cloud
    :param params: the parameters, if not given, it will be initialized with the default parameters
    :return: the cube with ones where there are at least one point and zero otherwise
    """
    def cartesian_to_spherical(x, y, z):
        """
        Generic cartesian to spherical.

        :param x: the first coordinate of the point
        :param y: the second coordinate of the point
        :param z: the third coordinate of the point
        :return: the range,az,el point in radians
        """
        range_ = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        azimuth = np.arctan2(y, x)  
        elevation = np.arcsin(np.clip(z / np.maximum(range_, 1e-8), -1.0, 1.0))
        azimuth = np.clip(azimuth, arrA[0] + 1e-6, arrA[-1] - 1e-6)
        
        return np.stack([range_, azimuth, elevation], axis=1)
    
    def non_uniform_voxelize_numpy(point_cloud, x_axis, y_axis, z_axis):
        """
        Voxelise a pointcloud in a non-uniform cube. The non-uniform axis are given as input to the function.
        In practise is used to voxelise the lidar point cloud into the radar non-uniform cube.

        :param point_cloud: the point cloud to voxelise
        :param x_axis: the first axis of the non-uniform cube
        :param y_axis: the second axis of the non-uniform cube
        :param z_axis: the third axis of the non-uniform cube
        :return: the cube with ones where there are at least one point and zero otherwise
        """
        num_x = len(x_axis)
        num_y = len(y_axis)
        num_z = len(z_axis)

        # Initialize the voxel grid with zeros as boolean tensors
        voxel_grid = np.zeros((num_x, num_y, num_z))
        #voxel_grid_count = np.zeros((num_x, num_y, num_z), dtype=np.uint16)

        # Calculate voxel indices for each axis using broadcasting
        x_indices = np.searchsorted(x_axis, point_cloud[..., 0], side='left')
        y_indices = np.searchsorted(y_axis, point_cloud[..., 1], side='left')
        z_indices = np.searchsorted(z_axis, point_cloud[..., 2], side='left')

        # Clip to pretend over
        # x_indices = np.clip(x_indices, 0, num_x - 1)
        # y_indices = np.clip(y_indices, 0, num_y - 1)
        # z_indices = np.clip(z_indices, 0, num_z - 1)

        # This is just in case there are some points outside the grid. It should not be the case since we clean first
        # the lidar point cloud.
        valid_indices = (x_indices > 1) & (x_indices < num_x-1) & (y_indices > 1) & (y_indices < num_y-1) & (
                z_indices > 1) & (z_indices < num_z-1)
        x_indices = x_indices[valid_indices]
        y_indices = y_indices[valid_indices]
        z_indices = z_indices[valid_indices]
        point_cloud = point_cloud[valid_indices, :]

        # for xi, yi, zi in zip(x_indices, y_indices, z_indices):
        #     voxel_grid_count[xi, yi, zi] += 1

        # voxel_grid = (voxel_grid_count >= 3).astype(np.uint8)

        # Correct the indices, so they are the closest, and not always the left ones.
        # condition = (x_indices > 0) & ((x_indices == num_x) | (
        #         np.abs(point_cloud[..., 0] - x_axis[x_indices - 1]) < np.abs(point_cloud[..., 0] - x_axis[x_indices])))

        # x_indices[condition] = x_indices[condition] - 1

        # condition = (y_indices > 0) & ((y_indices == num_y) | (
        #         np.abs(point_cloud[..., 1] - y_axis[y_indices - 1]) < np.abs(point_cloud[..., 1] - y_axis[y_indices])))

        # y_indices[condition] = y_indices[condition] - 1

        # condition = (z_indices > 0) & ((z_indices == num_z) | (
        #         np.abs(point_cloud[..., 2] - z_axis[z_indices - 1]) < np.abs(point_cloud[..., 2] - z_axis[z_indices])))

        # z_indices[condition] = z_indices[condition] - 1

        # Mark the voxel as occupied using the mask
        voxel_grid[x_indices, y_indices, z_indices] = 1

        return voxel_grid
    
    spher = cartesian_to_spherical(lidar_pc[:, 0], lidar_pc[:, 1], lidar_pc[:, 2])

    lidar_cube = non_uniform_voxelize_numpy(spher, arrR, arrA, arrE)
    indices = lidar_cube.nonzero()
    torch_style_indices = np.stack(indices, axis=-1)
    indices_list = torch_style_indices.tolist()

    return indices_list


def compute_flow_in_cartesian(label, arrR, arrA, arrE):
    def label2cube(label, coords, arrR, arrA, arrE):
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
        # init
        r_dim, a_dim, e_dim = len(arrR), len(arrA), len(arrE)
        r_bin, a_bin, e_bin = np.mean(arrR[1:]-arrR[:-1]), np.mean(arrA[1:]-arrA[:-1]), np.mean(arrE[1:]-arrE[:-1])
        bin_tensor = torch.tensor([r_bin, a_bin, e_bin], dtype=torch.float).view(3, 1, 1, 1)
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
        return flows
    def pseudolabel(label, arrR, arrA, arrE):
        # init
        r_dim, a_dim, e_dim = len(arrR), len(arrA), len(arrE)
        pcl_label_restored = torch.zeros((r_dim, a_dim, e_dim), dtype=torch.int8)
        pcllist = label['pcllist']
        indices_tensor = torch.tensor(pcllist)
        pcl_label_restored[indices_tensor[:, 0], indices_tensor[:, 1], indices_tensor[:, 2]] = 1
        negative_tensor = torch.zeros_like(pcl_label_restored)
        pcl_label_restored = torch.where(pcl_label_restored == 1, pcl_label_restored, negative_tensor)
        pcl_label_restored = pcl_label_restored.unsqueeze(0)
        pcl_label = pcl_label_restored.float()
        return pcl_label
    def get_coords(arrR, arrA, arrE):
        r_min, r_max, a_min, a_max, e_min, e_max = arrR.min(), arrR.max(), arrA.min(), arrA.max(), arrE.min(), arrE.max()
        # rae dimension
        r_dim, a_dim, e_dim = len(arrR), len(arrA), len(arrE)
        r_coords = torch.linspace(r_min, r_max, r_dim).view(r_dim, 1, 1).expand(r_dim, a_dim, e_dim)
        a_coords = torch.linspace(a_min, a_max, a_dim).view(1, a_dim, 1).expand(r_dim, a_dim, e_dim)
        e_coords = torch.linspace(e_min, e_max, e_dim).view(1, 1, e_dim).expand(r_dim, a_dim, e_dim)
        coords = torch.stack([r_coords, a_coords, e_coords], dim=0)
        coords = coords.to(dtype=torch.float)
        return coords
    def spherical_to_cartesian(spherical_points):
        range_, azimuth, elevation = spherical_points[:, 0], spherical_points[:, 1], spherical_points[:, 2]
        # Calculate x, y, z from spherical coordinates
        x = range_ * torch.cos(elevation) * torch.cos(azimuth)
        y = range_ * torch.cos(elevation) * torch.sin(azimuth)
        z = range_ * torch.sin(elevation)
        # Stack the Cartesian coordinates into a tensor of shape (N, 3)
        return torch.stack([x, y, z], dim=1)
    r_bin, a_bin, e_bin = np.mean(arrR[1:]-arrR[:-1]), np.mean(arrA[1:]-arrA[:-1]), np.mean(arrE[1:]-arrE[:-1])
    device = 'cpu'
    coords = get_coords(arrR, arrA, arrE)
    pcl_label = pseudolabel(label, arrR, arrA, arrE)
    flow_label = label2cube(label, coords, arrR, arrA, arrE)
    bin_tensor = torch.tensor([r_bin, a_bin, e_bin], dtype=torch.float32, device=device).view(3, 1, 1, 1)
    flow_label = flow_label * bin_tensor
    # take gt point
    mask_gt = (pcl_label > 0.9).float()
    mask_gt = mask_gt.bool()
    mgt = mask_gt[0] # (R, A, E)
    gtpcl_coord = coords[:, mgt].T  # (3, N) to (N, 3)
    gtpcl_coord = spherical_to_cartesian(gtpcl_coord)
    # flow 
    gtf = flow_label 
    gtf = gtf[:, mgt].T
    # Please note the flowlabel is mean (dr, da, de) in Spherical Coordinate
    c = coords[:, mgt].T
    valid_flow_warpgt = c + gtf
    valid_coords_in_Cartesiangt = spherical_to_cartesian(c)
    valid_flow_warp_in_Cartesiangt = spherical_to_cartesian(valid_flow_warpgt)
    valid_flow_cartesiangt = valid_flow_warp_in_Cartesiangt - valid_coords_in_Cartesiangt
    return gtpcl_coord.cpu().numpy(), valid_flow_cartesiangt.cpu().numpy()


def compute_flow_in_lidar(lidar_point, label, arrR, arrA, arrE):
    
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
        
    # roi filter
    lidar_point_polar = cartesian_to_spherical(lidar_point[:, 0], lidar_point[:, 1], lidar_point[:, 2])
    lidar_point_polar = roi_filter(lidar_point_polar, arrR, arrA, arrE)
    lidar_point = spherical_to_cartesian(lidar_point_polar)
    # flow
    N = lidar_point.shape[0]
    flows = np.zeros_like(lidar_point)  # (N, 3)
    mask_bg = np.full((N,), True, dtype=bool)
    t_rigid = np.array(label['gt_rigid_trans'], dtype=np.float32)  # (4, 4)
    flows = transform_and_convert(lidar_point, mask_bg, t_rigid, flows)
    # fg trans
    for fg_trans in label['gt_fg_rigid_trans']:
        bbox_in_rae, t_ego_bbx, _,_,_,_,_ = fg_trans
        r_minbbox, r_maxbbox, a_minbbox, a_maxbbox, e_minbbox, e_maxbbox = bbox_in_rae
        print('bbox in rae', bbox_in_rae)
        mask_fg = (lidar_point_polar[:,0] >= r_minbbox) & (lidar_point_polar[:,0] <= r_maxbbox) & \
                    (lidar_point_polar[:,1] >= a_minbbox) & (lidar_point_polar[:,1] <= a_maxbbox) & \
                    (lidar_point_polar[:,2] >= e_minbbox) & (lidar_point_polar[:,2] <= e_maxbbox)
        print('mask fg num', np.count_nonzero(mask_fg))
        print('all num', N)
        t_ego_bbx = np.array(t_ego_bbx, dtype=np.float32)
        flows = transform_and_convert(lidar_point, mask_fg, t_ego_bbx, flows)

    return lidar_point, flows

def coordinate_generate(arrR, arrA, arrE):

    r_min, r_max, a_min, a_max, e_min, e_max = arrR.min(), arrR.max(), arrA.min(), arrA.max(), arrE.min(), arrE.max()
    # rae dimension
    r_dim, a_dim, e_dim = len(arrR), len(arrA), len(arrE)
    r_coords = torch.linspace(r_min, r_max, r_dim).view(r_dim, 1, 1).expand(r_dim, a_dim, e_dim)
    a_coords = torch.linspace(a_min, a_max, a_dim).view(1, a_dim, 1).expand(r_dim, a_dim, e_dim)
    e_coords = torch.linspace(e_min, e_max, e_dim).view(1, 1, e_dim).expand(r_dim, a_dim, e_dim)
    coords = torch.stack([r_coords, a_coords, e_coords], dim=0) # 3 R A E
    coords = coords.to(dtype=torch.float)
    return coords

def load_physical_values(is_in_rad=True, is_with_doppler=True):
        def center_downsample(arr, target_len):
            center_idx = np.argmin(np.abs(arr))
            num_each_side = (target_len - 1) // 2
            indices = [center_idx]
            left = center_idx - 2
            right = center_idx + 2
            while len(indices) < target_len:
                if left >= 0:
                    indices.insert(0, left)
                if len(indices) < target_len and right < len(arr):
                    indices.append(right)
                left -= 2
                right += 2
                if left < 0 and right >= len(arr):
                    break 
            indices = sorted(indices)
            return arr[indices]
        
        temp_values = scipy.io.loadmat('resources/info_arr.mat')
        arr_range = temp_values['arrRange']
        if is_in_rad:
            deg2rad = np.pi/180.
            arr_azimuth = temp_values['arrAzimuth']*deg2rad
            arr_elevation = temp_values['arrElevation']*deg2rad
        else:
            arr_azimuth = temp_values['arrAzimuth']
            arr_elevation = temp_values['arrElevation']
        _, num_0 = arr_range.shape
        _, num_1 = arr_azimuth.shape
        _, num_2 = arr_elevation.shape
        arr_range = arr_range.reshape((num_0,))
        arr_azimuth = arr_azimuth.reshape((num_1,))
        arr_elevation = arr_elevation.reshape((num_2,))
        arr_azimuth = np.flip(-arr_azimuth)
        arr_elevation = np.flip(-arr_elevation)
        a_start = (num_1 - 96) // 2
        e_start = (num_2 - 32) // 2
        arr_azimuth = arr_azimuth[a_start:a_start+96]
        arr_elevation = arr_elevation[e_start:e_start+32]
        arr_range = arr_range[:128]
        arr_azimuth = center_downsample(arr_azimuth, 48)
        # arr_elevation = center_downsample(arr_elevation, 16)
        if is_with_doppler:
            arr_doppler = scipy.io.loadmat('resources/arr_doppler.mat')['arr_doppler']
            _, num_3 = arr_doppler.shape
            arr_doppler = arr_doppler.reshape((num_3,))
            return arr_range, arr_azimuth, arr_elevation, arr_doppler
        else:
            return arr_range, arr_azimuth, arr_elevation
        
def check_points_within_radius_kdtree(a, l_np, radius=0.5, min_points=3):
    """
    a: (3, R, A, E) torch tensor on any device
    l_np: (N, 3) numpy array, LiDAR point cloud
    Returns:
        result: (1, R, A, E) torch tensor, float32 (0 or 1)
    """
    device = a.device
    range_, azim, elev = a[0], a[1], a[2]  # shape: (R, A, E)
    # polar → xyz
    x = range_.cpu().numpy() * np.cos(elev.cpu().numpy()) * np.cos(azim.cpu().numpy())
    y = range_.cpu().numpy() * np.cos(elev.cpu().numpy()) * np.sin(azim.cpu().numpy())
    z = range_.cpu().numpy() * np.sin(elev.cpu().numpy())
    coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # (R*A*E, 3)
    # kdtree
    tree = cKDTree(l_np)  # l_np 是 (N, 3) numpy array
    neighbors = tree.query_ball_point(coords, r=radius)  # list of lists
    # neighbors
    flags = np.array([len(nbs) >= min_points for nbs in neighbors], dtype=np.float32)  # (R*A*E,)
    flags = torch.from_numpy(flags).to(device).reshape(1, *range_.shape)  # (1, R, A, E)
    return flags

def chamfer_distance(pcl1: np.ndarray, pcl2: np.ndarray):
    """
    Compute Chamfer Distance between two point clouds using NumPy.

    :param pcl1: np.ndarray of shape (N, 3)
    :param pcl2: np.ndarray of shape (M, 3)
    :return: Chamfer Distance (float)
    """
    assert pcl1.ndim == 2 and pcl1.shape[1] == 3, "pcl1 must be of shape (N, 3)"
    assert pcl2.ndim == 2 and pcl2.shape[1] == 3, "pcl2 must be of shape (M, 3)"
    # kdtree
    tree1 = cKDTree(pcl1)
    tree2 = cKDTree(pcl2)
    dist1, _ = tree2.query(pcl1, k=1)
    dist2, _ = tree1.query(pcl2, k=1)
    chamfer = np.mean(dist1) + np.mean(dist2)
    return chamfer

def compute_detected_snr_stats_batch(echo_tensor: torch.Tensor,
                                     target_mask: torch.Tensor,
                                     ref_window_size: int = 5,
                                     guard_size: int = 0):
        device = echo_tensor.device
        eps = 1e-6

        # 
        window = torch.ones((1, 1, ref_window_size, ref_window_size, ref_window_size), device=device)
        center = ref_window_size // 2
        window[:, :, center - guard_size//2:center + guard_size//2 + 1,
                center - guard_size//2:center + guard_size//2 + 1,
                center - guard_size//2:center + guard_size//2 + 1] = 0
        num_ref_cells = window.sum()

        # 
        ref_energy = F.conv3d(
            echo_tensor, window,
            padding=ref_window_size // 2
        ) / (num_ref_cells + eps)

        # 
        detector_mask = target_mask

        target_power = echo_tensor[detector_mask] + eps
        ref_power = ref_energy[detector_mask] + eps
        snr_linear = target_power/2.302585 - ref_power/2.302585
        snr_db = 10 * snr_linear

        return snr_db.mean(), snr_db.std()

def compute_topk_mask_per_r_np(x1, k=10, r_start=10):
    """
    x1: numpy array of shape (C, R, A, E)
    returns: mask of shape (1, R, A, E)
    """
    C, R, A, E = x1.shape
    x1_rae = np.mean(x1, axis=0, keepdims=True)  # shape: (1, R, A, E)
    pseudo_mask = np.zeros_like(x1_rae)

    for r in range(r_start, R):
        # x_r shape: (1, A, E)
        x_r = x1_rae[:, r, :, :]  # shape: (1, A, E)
        x_r_flat = x_r.reshape(-1)
        # topk 
        topk_idx = np.argpartition(-x_r_flat, k)[:k]
        # mask
        mask_r_flat = np.zeros_like(x_r_flat)
        mask_r_flat[topk_idx] = 1.0
        mask_r = mask_r_flat.reshape(1, A, E)
        pseudo_mask[:, r, :, :] = mask_r

    return pseudo_mask.squeeze(0)      
