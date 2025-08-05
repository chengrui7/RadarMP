import numpy as np
import scipy.io
import os
import json

def save_pointcloud_to_bin(points: np.ndarray, filepath: str):
    assert points.ndim == 2 and points.shape[1] == 4
    points.astype(np.float32).tofile(filepath)
    print(f"lidar save to {filepath}, {points.shape[0]} point")

def save_calib_file(K_cam, Tr_velo_to_cam, R0_rect=None, output_path='calib.txt'):
    assert K_cam.shape == (4, 4)
    assert Tr_velo_to_cam.shape == (4, 4)
    if R0_rect is None:
        R0_rect = np.eye(3)
    K_cam34 = K_cam[:3, :]
    with open(output_path, 'w') as f:
        for i in range(4):
            flat_P = ' '.join([f'{v:.6f}' for v in K_cam34.flatten()])
            f.write(f'P{i}: {flat_P}\n')

        flat_R0 = ' '.join([f'{v:.1f}' for v in R0_rect.flatten()])
        f.write(f'R0_rect: {flat_R0}\n')

        # 3x4 Tr
        Tr = Tr_velo_to_cam[:3, :]
        flat_Tr = ' '.join([f'{v:.6f}' for v in Tr.flatten()])
        f.write(f'Tr_velo_to_cam: {flat_Tr}\n')
        f.write(f'Tr_imu_to_velo:\n')

    print(f"calib save to : {output_path}")

def trans_L2O_O2C(posel2o, posel2c):
    def inv_se3(T):
        Tinv = np.eye(4)
        Tinv[:3, :3] = T[:3, :3].T  # Transpose of the rotation matrix
        Tinv[:3, 3] = -np.dot(Tinv[:3, :3], T[:3, 3])  # -R^T * t
        return Tinv
    poseo2l = inv_se3(posel2o)
    poseo2c = posel2c @ poseo2l
    return poseo2c

def save_transform_to_json(matrix: np.ndarray, output_path: str):
    assert matrix.shape == (4, 4)
    flat_list = matrix.flatten().tolist()
    keys=["odomToCamera", "mapToCamera", "UTMToCamera"]
    with open(output_path, 'w') as f:
        for key in keys:
            json_line = json.dumps({key: flat_list})
            f.write(json_line + '\n')
    print(f"save T to json: {output_path}")

def save_label_file(label, output_path: str):
    with open(output_path, 'w') as f:
        for uqi_obj, uqi_obj3d in zip(label.tracking2d, label.tracking):
            cls_name, bbox_param, idx_obj, idx_now, unique_id, bgr, rgb, velocity, corners_rotated, lines, bbox2d, alpha = uqi_obj
            _, bbox_param3d, _, _,_,_,_,_,_,_,_,_ = uqi_obj3d
            u,v,w,h = bbox2d
            x,y,z,theta,l,w,h = bbox_param3d
            left = u;top = v;right = u + w - 1;bottom = v + h - 1
            f.write(f'{cls_name} {unique_id} 0 {alpha} {left} {top} {right} {bottom} {h} {w} {l} {x} {y} {z} {theta} 1\n')

    print(f"save label to: {output_path}")