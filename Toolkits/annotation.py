import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

# config settings
seq = ['1','30','35','21', '7', '16', '44', '42']  #  '1','5','30','35','21', '7', '16', '44', '42'
for eseq in seq:
    dict_cfg = dict(
        path_data = dict(
            head = '/home/ruiqi/data/K-Radar/K-Radar/generated_files/' + eseq + '/',
            cam_path = 'cam-front',
            lidar_path = 'os2-64',
            radar_path = 'radar_tesseract',
            calib_path = 'info_calib/calib_radar_lidar.txt',
            odom_path = f'resources/odometry/gt/gt_{int(eseq):02d}.txt',
            label_path = 'info_label',
            vis_label_path = 'label_vis',
            vis_flow_path = 'flow_vis',
        ),
        calib = dict(z_offset=0.7),
        lidarroi = {
                'x':[0, 200.0],
                'y': [-100.0, 100.0],
                'z': [-0.25, 30],
                }
    )
    dict_meta = Datadict.GetDict(dict_cfg)
    #assert len(dict_meta.dict_item['meta']['lidar']) == dict_meta.dict_item['odom'].shape[0], \
        #f"Length mismatch: len(dict.dict_item['meta']['lidar']) = {len(dict_meta.dict_item['meta']['lidar'])}, " \
        #f"dict.dict_item['odom'].shape[0] = {dict_meta.dict_item['odom'].shape[0]}"

    dict_flow = dict(
        flow = {},
        x_max = np.finfo(np.float32).tiny,
        y_max = np.finfo(np.float32).tiny
    )

    # output path
    output_dir = dict_cfg['path_data']['head']+"info_label_sceneflow"
    os.makedirs(output_dir, exist_ok=True)

    for frame_number,_ in enumerate(dict_meta.label_item):

        if frame_number + 1 >= len(dict_meta.label_item):
            break

        print(dict_meta.label_item[frame_number])

        print(dict_meta.dict_item['meta']['lidar'][dict_meta.label_item[frame_number]['lidar_frame']])

        lidar_pcd = Dataloader.PointCloudPcd(dict_meta.dict_item['meta']['lidar'][dict_meta.label_item[frame_number]['lidar_frame']])
        lidar_pcd.calib_xyz(dict_meta.dict_item['calib'][-3:])
        lidar_pcd.roi_filter(dict_cfg['lidarroi'])
        # radar_cube = Dataloader.RadarTesseract(dict_meta.dict_item['meta']['radar'][dict_meta.label_item[frame_number]['radar_frame']])
        # next_radar_cube = Dataloader.RadarTesseract(dict_meta.dict_item['meta']['radar'][dict_meta.label_item[frame_number + 1]['radar_frame']])
        label = Dataloader.Label3dBbox(dict_meta.label_item[frame_number], dict_meta.dict_item['calib'][-3:])

        # fig1 = plt.figure(figsize=(24, 5))
        # ax1 = fig1.add_subplot(131, projection='3d')
        # ax1.scatter(lidar_pcd.roipoints[:, 0], lidar_pcd.roipoints[:, 1], lidar_pcd.roipoints[:,2], c=lidar_pcd.roipoints[:,2], cmap='GnBu', alpha=0.8, s=1)
        # ax1.view_init(elev=30, azim=-180)
        # ax1.set_box_aspect([2, 2, 0.4])
        # ax1.set_xlabel('x (m)')
        # ax1.set_ylabel('y (m)')
        # ax1.set_zlabel('z (m)')
        # for uqi_obj in label.tracking:
        #     cls_name, _, _, _,unique_id,_,rgb,velocity,corners_rotated,lines,_,_ = uqi_obj
        #     for line in lines:
        #         ax1.plot3D(*zip(corners_rotated[line[0]], corners_rotated[line[1]]), color=rgb)
        # ax2 = fig1.add_subplot(132)
        # c = ax2.pcolormesh(radar_cube.bev_arry, radar_cube.bev_arrx, radar_cube.bevxy, cmap='jet')
        # for uqi_obj in label.tracking:
        #     cls_name, _, _, _,unique_id,_,rgb,velocity,_,_,bevcorners,bevlines = uqi_obj
        #     for bevline in bevlines:
        #         ax2.plot(*zip([bevcorners[bevline[0]][1], bevcorners[bevline[0]][0]], 
        #                     [bevcorners[bevline[1]][1], bevcorners[bevline[1]][0]]), color=rgb)
        # ax2.invert_xaxis()
        # ax3 = fig1.add_subplot(133)
        # img = plt.imread(dict_meta.dict_item['meta']['cam'][dict_meta.label_item[frame_number]['cam_frame']])
        # img = img[:, :img.shape[1] // 2]
        # ax3.imshow(img, aspect='auto')
        # ax3.axis('off')
        # fig1.savefig(dict_cfg['path_data']['vis_label_path']+'/label_'+dict_meta.label_item[frame_number]['lidar_frame']+'.png', dpi=150, bbox_inches='tight')

        next_lidar_pcd = Dataloader.PointCloudPcd(dict_meta.dict_item['meta']['lidar'][dict_meta.label_item[frame_number+1]['lidar_frame']])
        next_lidar_pcd.calib_xyz(dict_meta.dict_item['calib'][-3:])
        next_lidar_pcd.roi_filter(dict_cfg['lidarroi'])

        # check odom
        pose0 = dict_meta.dict_item['odom'][int(dict_meta.label_item[frame_number]['lidar_frame']) - int(dict_meta.label_item[0]['lidar_frame'])]
        pose1 = dict_meta.dict_item['odom'][int(dict_meta.label_item[frame_number+1]['lidar_frame']) - int(dict_meta.label_item[0]['lidar_frame'])]
        print(pose0)
        print(pose1)
        relative_pose = Dataloader.RelativePose(pose0, pose1, dict_meta.dict_item['calib'][-3:])
        print(relative_pose.pose0to1_calib)

        # use odom transform lidar
        chamfer_dist0 = Annoutil.chamfer_dist(lidar_pcd.roipoints, next_lidar_pcd.roipoints) 
        print(chamfer_dist0)
        rigid_flow = Annoutil.get_rigid_flow(lidar_pcd.roipoints, relative_pose.pose0to1_calib)
        transpoints = lidar_pcd.roipoints + rigid_flow
        chamfer_dist1 = Annoutil.chamfer_dist(transpoints, next_lidar_pcd.roipoints) 
        print(chamfer_dist1)

        # fig2 = plt.figure(figsize=(16, 5))
        # ax4 = fig2.add_subplot(121, projection='3d')
        # ax4.scatter(lidar_pcd.roipoints[:, 0], lidar_pcd.roipoints[:, 1], lidar_pcd.roipoints[:,2], c='r', alpha=0.5, s=1)
        # ax4.scatter(next_lidar_pcd.roipoints[:, 0], next_lidar_pcd.roipoints[:, 1], next_lidar_pcd.roipoints[:,2], c='b', alpha=0.5, s=1)
        # ax4.view_init(elev=35, azim=-180)
        # ax4.set_box_aspect([2, 2, 0.4])
        # ax4.set_xlabel('x (m)')
        # ax4.set_ylabel('y (m)')
        # ax4.set_zlabel('z (m)')
        # ax5 = fig2.add_subplot(122, projection='3d')
        # ax5.scatter(transpoints[:, 0], transpoints[:, 1], transpoints[:,2], c='r', alpha=0.5, s=1)
        # ax5.scatter(next_lidar_pcd.roipoints[:, 0], next_lidar_pcd.roipoints[:, 1], next_lidar_pcd.roipoints[:,2], c='b', alpha=0.5, s=1)
        # ax5.view_init(elev=35, azim=-180)
        # ax5.set_box_aspect([2, 2, 0.4])
        # ax5.set_xlabel('x (m)')
        # ax5.set_ylabel('y (m)')
        # ax5.set_zlabel('z (m)')

        # get fg_flow and move segment
        label1 = label
        label2 = Dataloader.Label3dBbox(dict_meta.label_item[frame_number+1], dict_meta.dict_item['calib'][-3:])
        fg_flows, fg_idx, fg_trans = Annoutil.get_fg_rigid_flow(label1, label2, lidar_pcd.roipoints, next_lidar_pcd.roipoints, dict_meta.info_rae)
        # get fg relative flow
        flow_nr = fg_flows[fg_idx] - rigid_flow[fg_idx]
        # obtain the index for moving points from foreground
        mov_idx = np.array(fg_idx)[np.linalg.norm(flow_nr,axis=1)>0.05]
        if len(mov_idx)>0:
            static_idx = np.delete(np.arange(0, lidar_pcd.roipoints.shape[0]), mov_idx)
        else:
            static_idx = np.arange(0, lidar_pcd.roipoints.shape[0])

        gt_movsegment = np.zeros(lidar_pcd.roipoints.shape[0], dtype=np.float32)
        gt_sceneflow = np.zeros((lidar_pcd.roipoints.shape[0],3), dtype=np.float32)

        gt_movsegment[static_idx] = 1
        gt_sceneflow[static_idx] = rigid_flow[static_idx]
        if len(mov_idx)>0:
            gt_movsegment[mov_idx] = 0
            gt_sceneflow[mov_idx] = fg_flows[mov_idx]
        gt_movsegment = gt_movsegment.astype(bool)

        # label_dict_frame = dict(
        #     gt_rigid_trans = relative_pose.pose0to1_calib,
        #     gt_fg_rigid_trans = fg_trans,
        #     gt_movsegment = gt_movsegment,
        #     #points = lidar_pcd.roipoints,
        #     frame = frame_number
        # )
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #coordsframe, flowsframe = Annoutil.label2cube(radar_cube, label_dict_frame)
        #pcllist = Annoutil.get_pseudolabel_pcl(radar_cube.cube, next_radar_cube.cube, coordsframe, flowsframe, device)

        pcllist = Annoutil.lidar2labelpcl(lidar_pcd.roipoints, dict_meta.arr_range, dict_meta.arr_azimuth, dict_meta.arr_elevation)
        nextpcllist = Annoutil.lidar2labelpcl(next_lidar_pcd.roipoints, dict_meta.arr_range, dict_meta.arr_azimuth, dict_meta.arr_elevation)
        #print(pcllist)
        dict_flow['flow'][frame_number] = dict(
            gt_rigid_trans = relative_pose.pose0to1_calib,
            gt_fg_rigid_trans = fg_trans,
            gt_movsegment = gt_movsegment,
            pcllist = pcllist,
            nextpcllist = nextpcllist, 
            frame = frame_number
        )
        dict_flow['x_max'] = max(dict_flow['x_max'], np.max(np.abs(gt_sceneflow[:, 0])))
        dict_flow['y_max'] = max(dict_flow['x_max'], np.max(np.abs(gt_sceneflow[:, 1])))
        # plt.close('all')

        file_path = os.path.join(output_dir, 'sceneflowgt_'+dict_meta.label_item[frame_number]['lidar_frame']+'.npy')
        np.save(file_path, dict_flow['flow'][frame_number])
        print(f"label frame saved to directory: {file_path}")

    print(f"Flow data saved to directory: {output_dir}")


# for frame_number, flow_data in dict_flow['flow'].items():

#     gt_movsegment = flow_data['gt_movsegment']
#     gt_sceneflow = flow_data['gt_sceneflow']
#     points = flow_data['points']
#     frame = flow_data['label_frame']

#     # plot flow and mov segment
#     # Visualize Motion Segment
#     fig3 = plt.figure(figsize=(16, 12))
#     ax31 = fig3.add_subplot(121, projection='3d')
    
#     mov_point0 = points[~gt_movsegment]
#     stat_point0 = points[gt_movsegment]

#     ax31.scatter(mov_point0[:, 0], mov_point0[:, 1], mov_point0[:,2], color=[1,0.7059,0.3922], label="dynamic", alpha=0.5, s=1)
#     ax31.scatter(stat_point0[:, 0], stat_point0[:, 1], stat_point0[:,2], color=[0.3922,0.7059,1], label="static", alpha=0.5, s=1)
#     ax31.view_init(elev=35, azim=-180)
#     ax31.set_box_aspect([2, 2, 0.4])
#     ax31.set_xlabel('x (m)')
#     ax31.set_ylabel('y (m)')
#     ax31.set_zlabel('z (m)')
#     ax31.legend(markerscale=5)

#     # hsv to 3d color wheel
#     cw_res = 500
#     cwx = np.linspace(-np.max(dict_flow['x_max']), np.max(dict_flow['x_max']), cw_res)
#     cwy = np.linspace(-np.max(dict_flow['y_max']), np.max(dict_flow['y_max']), cw_res)
#     cwX, cwY = np.meshgrid(cwx, cwy)
#     cwR = np.sqrt(cwX**2 + cwY**2) 
#     cwTheta = np.arctan2(cwY, cwX)  
#     cwH = (cwTheta % (2 * np.pi)) / (2 * np.pi) 
#     cwS = cwR/1.5/np.max(cwR)                     
#     cwV = 1- cwR/4/np.max(cwR)                
#     cwHSV = np.stack([cwH, cwS, cwV], axis=-1)
#     cwRGB = hsv_to_rgb(cwHSV)

#     scene_flow_x = gt_sceneflow[:, 0]
#     scene_flow_y = gt_sceneflow[:, 1]
#     magnitudes = np.sqrt(scene_flow_x**2 + scene_flow_y**2)
#     directions_xy = np.arctan2(scene_flow_y, scene_flow_x)  # 基于 xy 平面
#     sfH = (directions_xy % (2 * np.pi)) / (2 * np.pi)  #红绿蓝
#     sfS = magnitudes / 1.5 / np.max(cwR)   #彩色        
#     sfV = 1- magnitudes / 4 / np.max(cwR)    #亮度                         
#     sfHSV = np.stack([sfH, sfS, sfV], axis=-1)  # 合并 HSV
#     scene_flow_colors = hsv_to_rgb(sfHSV)
#     ax32 = fig3.add_subplot(122, projection='3d')
#     ax32.scatter(points[:,0], points[:,1], points[:,2], c=scene_flow_colors, s=1)
#     ax32.view_init(elev=35, azim=-180)
#     ax32.set_box_aspect([2, 2, 0.4])
#     ax32.set_xlabel('x (m)')
#     ax32.set_ylabel('y (m)')
#     ax32.set_zlabel('z (m)')

#     ax33  = fig3.add_axes([0.8, 0.6, 0.1, 0.1])  # 设置色轮位置和大小
#     ax33.imshow(cwRGB, extent=[-1, 1, -1, 1], origin='lower')
#     ax33.set_xticks([])  
#     ax33.set_yticks([])
#     ax33.set_title("Color Wheel")
#     fig3.savefig(dict_cfg['path_data']['vis_flow_path']+'/flow_'+dict_meta.label_item[frame]['lidar_frame']+'.png', dpi=150, bbox_inches='tight')
#     plt.close('all')