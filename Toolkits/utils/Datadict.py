import os
import os.path as osp
import numpy as np
import open3d as o3d

from scipy.io import loadmat 
from tqdm import tqdm
from easydict import EasyDict

# Labels and lasers require coordinate value 
# modifications based on calibration parameters.

class GetDict():
    def __init__(self, dict_cfg):
        
        cfg = EasyDict(dict_cfg)
        cfg_from_yaml = False
        self.cfg=cfg
        # Init for tracking
        self.list_unique_ids = []
        self.list_unique_center = []
        self.list_unique_timestamp = []

        self.list_unique_ids_in_screen = []
        self.list_unique_ids_l_in_screen = []
        self.list_label_ids_in_screen = []
        self.list_label_ids_l_in_screen = []

        b = np.random.choice(range(256), size=500).reshape(-1,1)
        g = np.random.choice(range(256), size=500).reshape(-1,1)
        r = np.random.choice(range(256), size=500).reshape(-1,1)

        bgr = np.concatenate([b,g,r], axis=1)
        bgr = list(map(lambda x: tuple(x), bgr.tolist()))
        self.list_unique_colors = list(set(bgr))
        self.label_item = self.load_label_item()

        # Init for data
        self.dict_item = self.load_dict_item()

        # Init for radar axiz
        self.arr_range, self.arr_azimuth, self.arr_elevation, self.arr_doppler = self.load_physical_values(is_with_doppler=True)
    
        arr_r = self.arr_range
        arr_a = self.arr_azimuth
        arr_e = self.arr_elevation

        r_min = np.min(arr_r)
        r_bin = np.mean(arr_r[1:]-arr_r[:-1])
        r_max = np.max(arr_r)
        
        a_min = np.min(arr_a)
        a_bin = np.mean(arr_a[1:]-arr_a[:-1])
        a_max = np.max(arr_a)

        e_min = np.min(arr_e)
        e_bin = np.mean(arr_e[1:]-arr_e[:-1])
        e_max = np.max(arr_e)

        self.info_rae = [
            [r_min, r_bin, r_max],
            [a_min, a_bin, a_max],
            [e_min, e_bin, e_max]]
        
        
    def load_physical_values(self, is_in_rad=True, is_with_doppler=False):
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
        
        temp_values = loadmat('resources/info_arr.mat')
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
            arr_doppler = loadmat('resources/arr_doppler.mat')['arr_doppler']
            _, num_3 = arr_doppler.shape
            arr_doppler = arr_doppler.reshape((num_3,))
            return arr_range, arr_azimuth, arr_elevation, arr_doppler
        else:
            return arr_range, arr_azimuth, arr_elevation
        
    
        

    def load_dict_item(self):
        def get_frames_and_paths(directory):
            """
            It generates a dictionary with paths and timstamps from a given directory.
            It is used to generate the dictionaries for training the neural network.
            It assumes the timestamp is in the name of the file in ROS format.

            :param directory: the directory to find the files
            :return: the dictionary with timestamps and file names
            """
            # Gets a dictionary with timestamps as keys and file paths as values from a directory.
            frames_paths = {}
            for filename in os.listdir(directory):
                if filename.endswith(".pcd") or filename.endswith(".png") or filename.endswith(".mat"):
                    frame = filename.split('_')[1].split('.')[0]
                    # Convert seconds and nanoseconds to a single integer timestamp (in nanoseconds)
                    file_path = os.path.join(directory, filename)
                    frames_paths[frame] = file_path
            print(f'{directory.split("/")[-1]} sequence length: {len(frames_paths)}')
            return frames_paths
        
        def get_calib_values(path_calib, z_offset):
            f = open(path_calib, 'r')
            lines = f.readlines()
            f.close()
            list_calib = list(map(lambda x: float(x), lines[1].split(',')))
            list_values = [list_calib[0], list_calib[1], list_calib[2], z_offset] # framediff, X, Y, Z
            print(f'{path_calib.split("/")[-1].split(".")[0]} calib ext radar to lidar: {list_values}')
            return list_values
        
        def get_odom(path_odom):
            with open(path_odom, 'r') as f:
                lines = f.readlines()
                poses = np.zeros((len(lines), 4, 4))

                for idx, line in enumerate(lines):
                    poses[idx] = np.array([float(element) for element in line.rstrip().split(' ')] + [0., 0., 0., 1.]).reshape(4, 4)
            print(f'odom {path_odom.split("/")[-1].split(".")[0]} sequence length: {poses.shape[0]}')
            return poses

        dict_item = {}
        dict_item['meta'] = {}
        dict_item['meta']['cam'] = get_frames_and_paths(self.cfg['path_data']['head']+self.cfg['path_data']['cam_path'])
        dict_item['meta']['lidar'] = get_frames_and_paths(self.cfg['path_data']['head']+self.cfg['path_data']['lidar_path'])
        dict_item['meta']['radar'] = get_frames_and_paths(self.cfg['path_data']['head']+self.cfg['path_data']['radar_path'])
        dict_item['odom'] = get_odom(self.cfg['path_data']['odom_path'])
        dict_item['calib'] = get_calib_values(self.cfg['path_data']['head']+self.cfg['path_data']['calib_path'], self.cfg['calib']['z_offset'])

        return dict_item
    
    def load_label_item(self):
        def get_tuple_object(line):
            '''
            * in : e.g., '*, 0, 0, Sedan, 3.8, 5.535, -1.0, -93.1155, 2.112, 1.0347, 0.95' --> There are labels like this too
            * out: tuple ('Sedan', [x, y, z, theta, l, w, h], idx_obj, idx_now)
            '''
            list_values = line.split(',')

            if list_values[0] != '*':
                return None, None, None, None

            list_values = list_values[1:]

            idx_now = int(list_values[0])
            idx_obj = int(list_values[1])
            cls_name = list_values[2][1:]

            x = float(list_values[3])
            y = float(list_values[4])
            z = float(list_values[5])
            theta = float(list_values[6])
            theta = theta*np.pi/180.
            l = 2*float(list_values[7])
            w = 2*float(list_values[8])
            h = 2*float(list_values[9])

            return (cls_name, [x,y,z,theta,l,w,h], idx_obj, idx_now)

        
        path_label = self.cfg['path_data']['head']+self.cfg['path_data']['label_path']
        list_label = sorted(os.listdir(path_label), key=lambda x : int(x.split('_')[1].split('.')[0]))

        num_files = len(list_label)  
        label = [{} for _ in range(num_files)]

        for idx, lalel_file in enumerate(list_label):
            #print(f'now: {lalel_file}')
            file = open(os.path.join(path_label, lalel_file), 'r')
            lines = file.readlines()
            header = lines[0]
            indices = ((header.split(',')[0]).split('=')[1]).split('_')
            timestamp = float(((header.split(',')[1]).split('=')[1]).rstrip('\n'))
            idx_rdr, idx_os64, idx_cam_front, idx_os128, _ = indices
            ### Read labels ###
            lines = lines[1:]

            list_objects = []
            for line in lines:
                cls_name, bbox_params, idx_obj, idx_now = get_tuple_object(line)
                if cls_name is None:
                    continue
                list_objects.append([cls_name, bbox_params, idx_obj, idx_now])
            file.close()
            # Get Tracking Info
            track_objects = []
            for uqi_object in list_objects:
                cls_name, bbox_params, idx_obj, idx_now = uqi_object
                # New Object
                if idx_obj not in self.list_label_ids_l_in_screen:
                    idx_obj = -1
                if idx_obj == -1:
                    unique_id = len(self.list_unique_ids)
                    self.list_unique_ids.append(unique_id)
                    self.list_unique_timestamp.append(timestamp)
                    self.list_unique_center.append([bbox_params[0],bbox_params[1],bbox_params[2]])
                    
                    self.list_unique_ids_in_screen.append(unique_id)
                    self.list_label_ids_in_screen.append(idx_now)

                    bgr = self.list_unique_colors[unique_id]
                    velocity = -1
                # Old Object
                else:
                    corr_idx = self.list_label_ids_l_in_screen.index(idx_obj)
                    unique_id = self.list_unique_ids_l_in_screen[corr_idx]

                    last_timestamp = self.list_unique_timestamp[unique_id]
                    last_center = self.list_unique_center[unique_id]
                    now_timestamp = timestamp
                    now_center = [bbox_params[0],bbox_params[1],bbox_params[2]]
                    bgr = self.list_unique_colors[unique_id]
                    dt = now_timestamp - last_timestamp
                    if dt == 0:
                        velocity = 0.0  
                    else:
                        velocity = np.sqrt((now_center[0] - last_center[0])**2 + (now_center[1] - last_center[1])**2) / dt

                    # km/h
                    velocity = np.around(velocity*3.6, 2)

                    # sign
                    l2_last = (last_center[0])**2 + (last_center[1])**2
                    l2_now = (now_center[0])**2 + (now_center[1])**2
                    if l2_last > l2_now:
                        velocity = -velocity

                    self.list_unique_ids_in_screen.append(unique_id)
                    self.list_label_ids_in_screen.append(idx_now)
                    self.list_unique_center[unique_id] = now_center
                    self.list_unique_timestamp[unique_id] = now_timestamp

                track_objects.append([cls_name, bbox_params, idx_obj, idx_now, unique_id, bgr, velocity])

            self.list_label_ids_l_in_screen = (self.list_label_ids_in_screen).copy()
            self.list_label_ids_in_screen = []
            self.list_unique_ids_l_in_screen = (self.list_unique_ids_in_screen).copy()
            self.list_unique_ids_in_screen = []

            #Save To Dict

            label[idx] = dict(
                lidar_frame = idx_os64,
                radar_frame = idx_rdr,
                cam_frame = idx_cam_front,
                list_objects = list_objects,
                track_objects = track_objects,
                timestamp = timestamp,
            )
        
        return label
        


    