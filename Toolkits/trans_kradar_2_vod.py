import os
import numpy as np
import scipy.io
import sys
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import hsv_to_rgb

import torch

from utils import Datadict
from utils import Dataloader
from utils import Annoutil
from utils import TransVOD

seq = ['21', '35']  # '5', '42' '21', '35' have transed
for eseq in seq:
    dict_cfg = dict(
        path_data = dict(
            head = '/home/data/K-Radar/K-Radar/generated_files/' + eseq + '/',
            cam_path = 'cam-front',
            lidar_path = 'os2-64',
            radar_path = 'radar_tesseract',
            calib_path = 'info_calib/calib_radar_lidar.txt',
            odom_path = f'resources/odometry/gt/gt_{int(eseq):02d}.txt',
            label_path = 'info_label',
            vis_label_path = 'label_vis',
            vis_flow_path = 'flow_vis',
            save_path = '/home/data/K-RadarTemp/K-RadarTemp/' + eseq + '/Vod/',
        ),
        calib = dict(z_offset=0.7),
        lidarroi = {
                'x':[0, 200.0],
                'y': [-100.0, 100.0],
                'z': [-0.25, 30],
                }
    )
    dict_meta = Datadict.GetDict(dict_cfg)
    save_lidar_dir = dict_cfg['path_data']['save_path'] + 'lidar'
    os.makedirs(save_lidar_dir, exist_ok=True)
    save_cam_dir = dict_cfg['path_data']['save_path'] + 'image'
    os.makedirs(save_cam_dir, exist_ok=True)
    save_calib_dir = dict_cfg['path_data']['save_path'] + 'calib'
    os.makedirs(save_calib_dir, exist_ok=True)
    save_pose_dir = dict_cfg['path_data']['save_path'] + 'pose'
    os.makedirs(save_pose_dir, exist_ok=True)
    save_label_dir = dict_cfg['path_data']['save_path'] + 'label'
    os.makedirs(save_label_dir, exist_ok=True)
    # recurrent
    for frame_number,_ in enumerate(dict_meta.label_item):
        if frame_number >= len(dict_meta.label_item):
            break
        lidar_pcd = Dataloader.PointCloudPcd(dict_meta.dict_item['meta']['lidar'][dict_meta.label_item[frame_number]['lidar_frame']])
        lidar_pcd.calib_xyz(dict_meta.dict_item['calib'][-3:])
        save_lidar_path = os.path.join(save_lidar_dir, dict_meta.label_item[frame_number]['lidar_frame']+'.bin')
        TransVOD.save_pointcloud_to_bin(lidar_pcd.values[:,:4], save_lidar_path)
        img = plt.imread(dict_meta.dict_item['meta']['cam'][dict_meta.label_item[frame_number]['cam_frame']])
        img = img[:, :img.shape[1] // 2]
        save_cam_path = os.path.join(save_cam_dir, dict_meta.label_item[frame_number]['lidar_frame']+'.jpg')
        plt.imsave(save_cam_path, img)
        T1 = np.load("resources/cam_calib/T_npy/T_cam2pix_front0.npy")
        T2 = np.load("resources/cam_calib/T_npy/T_ldr2cam_front0.npy")
        TR2L = np.eye(4)
        TR2L[:3, 3] = -np.array(dict_meta.dict_item['calib'][-3:])
        TR2C = T2 @ TR2L
        save_calib_path = os.path.join(save_calib_dir, dict_meta.label_item[frame_number]['lidar_frame']+'.txt')
        TransVOD.save_calib_file(K_cam=T1, Tr_velo_to_cam=TR2C, output_path=save_calib_path)
        pose = dict_meta.dict_item['odom'][int(dict_meta.label_item[frame_number]['lidar_frame']) - int(dict_meta.label_item[0]['lidar_frame'])]
        poseO2C = TransVOD.trans_L2O_O2C(pose, T2)
        save_pose_path = os.path.join(save_pose_dir, dict_meta.label_item[frame_number]['lidar_frame']+'.json')
        TransVOD.save_transform_to_json(poseO2C, save_pose_path)
        label = Dataloader.Label3dBbox(dict_meta.label_item[frame_number], dict_meta.dict_item['calib'][-3:])
        label.trans3d_2_2d(TR2C, T1[:3,:3], img)
        save_label_path = os.path.join(save_label_dir, dict_meta.label_item[frame_number]['lidar_frame']+'.txt')
        TransVOD.save_label_file(label, save_label_path)



