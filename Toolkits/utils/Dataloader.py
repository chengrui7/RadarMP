import numpy as np
# import cv2
import open3d as o3d 
import scipy.io

# Labels and lasers require coordinate value 
# modifications based on calibration parameters.

class PointCloudPcd():
    def __init__(self, path_pcd:str, len_header:int=11, ego_offset:float=1e-3)->object:
        f = open(path_pcd, 'r')
        lines = f.readlines()
        f.close()
        self.path_pcd = path_pcd

        list_header = lines[:len_header]
        list_values = lines[len_header:]
        list_values = list(map(lambda x: x.split(' '), list_values))
        values = np.array(list_values, dtype=np.float32)
        values = values[ # delete (0,0)
            np.where(
                (values[:,0]<-ego_offset) | (values[:,0]>ego_offset) |  # x
                (values[:,1]<-ego_offset) | (values[:,1]>ego_offset)    # y
            )]
        self.values = values
        self.roi_values = values
        self.list_attr = (list_header[2].rstrip('\n')).split(' ')[1:]
        self.is_calibrated = False
        self.is_roi_filtered = False

    def __repr__(self)->str:
        str_repr = f'total {len(self.values)}x{len(self.list_attr)} points, fields = {self.list_attr}'
        if self.is_calibrated:
            str_repr += ', calibrated'
        if self.is_roi_filtered:
            str_repr += ', roi filtered'
        return str_repr
    
    @property
    def points(self): # x, y, z
        return self.values[:,:3]
    
    @property
    def roipoints(self): # x, y, z
        return self.roi_values[:,:3]
    
    @property
    def points_w_attr(self):
        return self.values

    def _get_o3d_pcd(self)->o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        return pcd

    # def _get_bev_pcd(self, dict_render)->np.array:
    #     x_min, x_bin, x_max = dict_render['ROI_X']
    #     y_min, y_bin, y_max = dict_render['ROI_Y']

    #     hue_type = dict_render['HUE']
    #     val_type = dict_render['VAL']

    #     pts_w_attr = (self.points_w_attr.copy()).tolist()
    #     pts_w_attr = np.array(sorted(pts_w_attr,key=lambda x: x[2])) # sort via z

    #     arr_x = np.linspace(x_min, x_max-x_bin, num=int((x_max-x_min)/x_bin)) + x_bin/2.
    #     arr_y = np.linspace(y_min, y_max-y_bin, num=int((y_max-y_min)/y_bin)) + y_bin/2.
        
    #     xy_mesh_grid_hsv = np.full((len(arr_x), len(arr_y), 3), 0, dtype=np.int64)
    #     x_idx = np.clip(((pts_w_attr[:,0]-x_min)/x_bin+x_bin/2.).astype(np.int64),0,len(arr_x)-1)
    #     y_idx = np.clip(((pts_w_attr[:,1]-y_min)/y_bin+y_bin/2.).astype(np.int64),0,len(arr_y)-1)

    #     hue_min, hue_max = dict_render[f'{hue_type}_ROI']
    #     hue_val = np.clip((pts_w_attr[:,self.list_attr.index(hue_type)]-hue_min)/(hue_max-hue_min),0.1,0.9)

    #     val_min, val_max = dict_render[f'{val_type}_ROI']
    #     val_val = np.clip((pts_w_attr[:,self.list_attr.index(val_type)]-val_min)/(val_max-val_min),0.5,0.9)

    #     xy_mesh_grid_hsv[x_idx,y_idx,0] = (hue_val*127.).astype(np.int64)
    #     xy_mesh_grid_hsv[x_idx,y_idx,1] = 255 # Saturation
    #     xy_mesh_grid_hsv[x_idx,y_idx,2] = (val_val*255.).astype(np.int64)

    #     xy_mesh_grid_rgb_temp = cv2.cvtColor(xy_mesh_grid_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    #     dilation = dict_render['DILATION']
    #     xy_mesh_grid_rgb_temp = cv2.dilate(xy_mesh_grid_rgb_temp, kernel=(dilation,dilation))

    #     xy_mesh_grid_rgb = np.full_like(xy_mesh_grid_rgb_temp, fill_value=255, dtype=np.uint8)
    #     x_ind_valid, y_ind_valid = np.where(np.sum(xy_mesh_grid_rgb_temp, axis=2)>0)
    #     xy_mesh_grid_rgb[x_ind_valid,y_ind_valid,:] = xy_mesh_grid_rgb_temp[x_ind_valid,y_ind_valid,:]

    #     xy_mesh_grid_rgb = np.flip(xy_mesh_grid_rgb, axis=(0,1))

    #     return xy_mesh_grid_rgb

    def calib_xyz(self, list_calib_xyz:list):
        arr_calib_xyz = np.array(list_calib_xyz, dtype=self.values.dtype).reshape(1,3)
        arr_calib_xyz = arr_calib_xyz.repeat(repeats=len(self.values), axis=0)
        self.values[:,:3] += arr_calib_xyz
        self.is_calibrated=True

    def roi_filter(self, dict_roi:dict):
        '''
        dict_roi
            key: 'attr', value: [attr_min, attr_max]
        e.g., {'x': [0, 100]}
        '''
        values = self.values.copy()
        for temp_key, v in dict_roi.items():
            if not (temp_key in self.list_attr):
                print(f'* {temp_key} is not in attr')
                continue
            v_min, v_max = v
            idx = self.list_attr.index(temp_key)
            values = values[
                np.where(
                    (values[:,idx]>v_min) & (values[:,idx]<v_max)
                )]
        self.roi_values = values
        self.is_roi_filtered=True

    def render_in_o3d(self):
        o3d.visualization.draw_geometries([self._get_o3d_pcd()])

    # def render_in_bev(self, dict_render:dict):
    #     img_bev = self._get_bev_pcd(dict_render)
    #     cv2.imshow('LiDAR PCD (in BEV)', img_bev)
    #     cv2.waitKey(0)


class RadarTesseract():
    def __init__(self, path_tesseract:str, is_doppler_separated:bool=False)->object:
        arr_tesseract = scipy.io.loadmat(path_tesseract)['arrDREA']
        arr_tesseract = np.transpose(arr_tesseract, (0, 1, 3, 2))  #DREA -> DRAE
        self.is_reverse_ae = True
        arr_tesseract = np.flip(np.flip(arr_tesseract, axis=2), axis=3)
        a_start = (arr_tesseract.shape[2] - 96) // 2
        e_start = (arr_tesseract.shape[3] - 32) // 2
        arr_tesseract = arr_tesseract[:, :128, a_start:a_start+96, e_start:e_start+32]
        arr_tesseract = arr_tesseract.copy()
        D, R, A, E = arr_tesseract.shape
        assert R % 2 == 0 and A % 2 == 0 and E % 2 == 0, "RAE need not be oods"
        arr_tesseract = arr_tesseract.reshape(D, R//2, 2, A//2, 2, E//2, 2)  # 分块
        arr_tesseract = arr_tesseract.mean(axis=(2, 4, 6))
        if is_doppler_separated:
            arr_tesseract_t = np.transpose(arr_tesseract, [1,2,3,0])
            sum_doppler = np.sum(arr_tesseract_t, axis=3, keepdims=True)
            arr_tesseract_t_norm = arr_tesseract_t / sum_doppler
            arr_tesseract_t_expectation = np.sum(arr_tesseract_t_norm * np.array([i for i in range(64)]), axis=3)
            arr_tesseract_t_expectation_idx = np.clip(arr_tesseract_t_expectation.astype(np.uint8), 0, 63)
            cube_doppler = self.arr_doppler[arr_tesseract_t_expectation_idx] # R, A, E
            cube_power = np.mean(arr_tesseract, axis=0)
            self.cube = arr_tesseract  # R,A,E
            self.doppler = False
        else:
            self.cube = arr_tesseract
            self.doppler = True   #D,R,A,E
        self.arr_range, self.arr_azimuth, self.arr_elevation, self.arr_doppler = self.load_physical_values()
        # self.bevxy, self.bevyx, self.bev_arrx, self.bev_arry = self.get_bev_ra2xy()
        

    def load_physical_values(self, is_in_rad=True):
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
        a_start = (num_1 - 96) // 2
        e_start = (num_2 - 32) // 2
        arr_azimuth = arr_azimuth[a_start:a_start+96]
        arr_elevation = arr_elevation[e_start:e_start+32]
        arr_range = arr_range[:128]
        arr_azimuth = center_downsample(arr_azimuth, 48)
        # arr_elevation = center_downsample(arr_elevation, 16)
        if self.is_reverse_ae:
            arr_azimuth = np.flip(-arr_azimuth)
            arr_elevation = np.flip(-arr_elevation)

        arr_doppler = scipy.io.loadmat('resources/arr_doppler.mat')['arr_doppler']
        _, num_3 = arr_doppler.shape
        arr_doppler = arr_doppler.reshape((num_3,))
        return arr_range, arr_azimuth, arr_elevation, arr_doppler
    
    def get_bev_ra2xy(self)->np.array:
        def find_nearest_two(value, arr):
            '''
            * args
            *   value: float, value in arr
            *   arr: np.array
            * return
            *   idx0, idx1 if is_in_arr else -1
            '''
            arr_temp = arr - value
            arr_idx = np.argmin(np.abs(arr_temp))
            
            try:
                if arr_temp[arr_idx] < 0: # min is left
                    if arr_temp[arr_idx+1] < 0:
                        return -1, -1
                    return arr_idx, arr_idx+1
                elif arr_temp[arr_idx] >= 0:
                    if arr_temp[arr_idx-1] >= 0:
                        return -1, -1
                    return arr_idx-1, arr_idx
            except:
                return -1, -1
        rdr_bev = np.mean(np.mean(self.cube, axis=0), axis=2)
        print(rdr_bev.shape)
        ra = 10*np.log10(rdr_bev * 1e-13)
        roi_x = [0, 0.4, 100]
        roi_y = [-40, 0.4, 40]
        min_x, bin_x, max_x = roi_x
        min_y, bin_y, max_y = roi_y
        arr_x = np.linspace(min_x, max_x-bin_x, int((max_x-min_x)/bin_x))+bin_x/2.
        arr_y = np.linspace(min_y, max_y-bin_y, int((max_y-min_y)/bin_y))+bin_x/2.

        max_r = np.max(self.arr_range)
        min_r = np.min(self.arr_range)

        max_azi = np.max(self.arr_azimuth)
        min_azi = np.min(self.arr_azimuth)

        num_y = len(arr_y)
        num_x = len(arr_x)
        
        arr_xy = np.zeros((num_x, num_y), dtype=np.float32)
        arr_yx = np.zeros((num_y, num_x), dtype=np.float32)

        # Inverse warping
        for idx_y, y in enumerate(arr_y):
            for idx_x, x in enumerate(arr_x):
                # bilinear interpolation

                r = np.sqrt(x**2 + y**2)
                #azi = np.arctan2(-y,x) # for real physical azimuth
                azi = np.arctan2(y,x)
                
                if (r < min_r) or (r > max_r) or (azi < min_azi) or (azi > max_azi):
                    continue
                
                try:
                    idx_r_0, idx_r_1 = find_nearest_two(r, self.arr_range)
                    idx_a_0, idx_a_1 = find_nearest_two(azi, self.arr_azimuth)
                except:
                    continue

                if (idx_r_0 == -1) or (idx_r_1 == -1) or (idx_a_0 == -1) or (idx_a_1 == -1):
                    continue
                
                ra_00 = ra[idx_r_0,idx_a_0]
                ra_01 = ra[idx_r_0,idx_a_1]
                ra_10 = ra[idx_r_1,idx_a_0]
                ra_11 = ra[idx_r_1,idx_a_1]

                val = (ra_00*(self.arr_range[idx_r_1]-r)*(self.arr_azimuth[idx_a_1]-azi)\
                        +ra_01*(self.arr_range[idx_r_1]-r)*(azi-self.arr_azimuth[idx_a_0])\
                        +ra_10*(r-self.arr_range[idx_r_0])*(self.arr_azimuth[idx_a_1]-azi)\
                        +ra_11*(r-self.arr_range[idx_r_0])*(azi-self.arr_azimuth[idx_a_0]))\
                        /((self.arr_range[idx_r_1]-self.arr_range[idx_r_0])*(self.arr_azimuth[idx_a_1]-self.arr_azimuth[idx_a_0]))

                arr_xy[idx_x, idx_y] = val
                arr_yx[idx_y, idx_x] = val
        print(arr_xy.shape)
        return arr_xy, arr_yx, arr_x, arr_y
    
class Label3dBbox():
    def __init__(self, track_params:dict, calib, is_need_calib:bool=True)->object:
        self.time = track_params['timestamp']
        self.lidarframe = track_params['lidar_frame']
        self.radarframe = track_params['radar_frame']
        self.camframe = track_params['cam_frame']
        self.tracking = []
        self.tracking2d = []
        if is_need_calib:
            for unique_object in track_params['track_objects']:
                cls_name, bbox_params, idx_obj, idx_now, unique_id, bgr, velocity = unique_object
                x,y,z,theta,l,w,h = bbox_params
                dx, dy, dz = calib
                x = x + dx
                y = y + dy
                z = z + dz
                center = (x, y, z)
                R = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0],
                    [0,              0,             1]])
                corners = np.array([[l/2, w/2, h/2], [l/2, w/2, -h/2], [l/2, -w/2, h/2], [l/2, -w/2, -h/2],
                            [-l/2, w/2, h/2], [-l/2, w/2, -h/2], [-l/2, -w/2, h/2], [-l/2, -w/2, -h/2]])
                corners_rotated = np.dot(corners, R.T) + center
                lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                    [0, 4], [1, 5], [2, 6], [3, 7]]
                bev_corners_rotated = np.array([[corners_rotated[0][0],corners_rotated[0][1]],
                                                [corners_rotated[2][0],corners_rotated[2][1]],
                                                [corners_rotated[4][0],corners_rotated[4][1]],
                                                [corners_rotated[6][0],corners_rotated[6][1]]])
                bevlines = [[0, 1],[1, 3],[3, 2],[2, 0]]
                rgb = [bgr[2]/255.0, bgr[1]/255.0, bgr[0]/255.0]
                self.tracking.append([cls_name, [x,y,z,theta,l,w,h], idx_obj, idx_now, unique_id, 
                    bgr, rgb, velocity, corners_rotated, lines, bev_corners_rotated, bevlines])
                
    def trans3d_2_2d(self, Tcam, Kcam, img):
        def project_lidar_to_image(lidar_points, K, T, img):
            lidar_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))  # (N, 4)
            cam_points = (T @ lidar_hom.T).T  # (N, 4) @ (4, 4) -> (N, 4)
            cam_points = cam_points[cam_points[:, 2] > 0]
            pixels = (K @ cam_points[:, :3].T).T  # (3,3) @ (N,3)^T → (3,N)^T → (N,3)
            pixels[:, 0] /= pixels[:, 2]  # 归一化
            pixels[:, 1] /= pixels[:, 2]
            h, w = img.shape[:2]
            mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
            valid_pixels = pixels[mask]
            return valid_pixels

        def adjust_bbox2d_by_center(bbox2d, image_shape, strength=0.05):
            x0, y0, w, h = bbox2d
            box_cx = x0 + w / 2
            img_cx = image_shape[1] / 2
            offset_ratio = (box_cx - img_cx) / img_cx
            if box_cx>img_cx:
                strength=0.2
                dx = offset_ratio * strength * image_shape[1]
                new_x0 = x0 + dx
            elif box_cx<img_cx/2:
                dx = offset_ratio * strength * image_shape[1]
                new_x0 = x0 + dx
            else:
                new_x0 = x0
            return [new_x0, y0, w, h]
        
        for uqi_obj in self.tracking:
            cls_name, bbox_param, idx_obj, idx_now, unique_id, \
                    bgr, rgb, velocity, corners_rotated, lines, bev_corners_rotated, bevlines = uqi_obj
            obj_center_3d = np.mean(corners_rotated, axis=0)  # (3,)
            obj_center_hom = np.append(obj_center_3d, 1)  # [x, y, z, 1]
            obj_center_cam = (Tcam @ obj_center_hom)[:3]  # [x_cam, y_cam, z_cam]
            observation_angle = np.arctan2(obj_center_cam[1], obj_center_cam[0])  # [-π, π]
            valid_pixels = project_lidar_to_image(corners_rotated, Kcam, Tcam, img)
            if valid_pixels.shape[0] == 0:
                continue
            min_u, max_u = valid_pixels[:, 0].min(), valid_pixels[:, 0].max()
            min_v, max_v = valid_pixels[:, 1].min(), valid_pixels[:, 1].max()
            bbox2d = [min_u, min_v, max_u - min_u, max_v - min_v]
            bbox2d = adjust_bbox2d_by_center(bbox2d, img.shape)
            self.tracking2d.append([cls_name, bbox_param, idx_obj, idx_now, unique_id, bgr, rgb, velocity, 
                    corners_rotated, lines, bbox2d, observation_angle])
                
class RelativePose():
    def __init__(self, pose0, pose1, calib)->object:
        self.pose0 = pose0
        self.pose1 = pose1

        self.pose0to1 = self.inv_se3(pose1) @ pose0  # T0to1
        Tcalib = np.eye(4)
        Tcalib[:3, 3] = calib
        self.calib = Tcalib  # TLtoR (calib T lidar to radar)
        self.pose0to1_calib = Tcalib @ self.pose0to1 @ self.inv_se3(Tcalib) # T0to1 calib in Radar system
        
    def inv_se3(self, T):
        """
        Compute the inverse of an SE(3) transformation matrix.

        Parameters:
            T (numpy.ndarray): 4x4 SE(3) transformation matrix.

        Returns:
            numpy.ndarray: Inverse of the input SE(3) transformation matrix.
        """
        Tinv = np.eye(4)
        Tinv[:3, :3] = T[:3, :3].T  # Transpose of the rotation matrix
        Tinv[:3, 3] = -np.dot(Tinv[:3, :3], T[:3, 3])  # -R^T * t
        return Tinv

