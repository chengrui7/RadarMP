import os
import scipy.io
import numpy as np
import yaml
import torch
import random

# Get RadarCube Loader Dict
def load_radarcube(data_path, seq, cube_path, cube2d_path, label_path, calib_path, lidar_path):
    """
    Generate a dictionary mapping frame numbers to file paths for the current frame and the next frame.

    Parameters:
        cube_path (str): Directory path containing tesseract files.

    Returns:
        dict: A dictionary where keys are frame numbers, and values are sub-dictionaries containing 
              paths to current and next radar tesseract files.
    """
    frame_dict = {}
    for eseq in seq:
        # Loop every sequence
        frame_dict[eseq] = {}
        label_dir = data_path + eseq + '/' + label_path
        cube_dir = data_path + eseq + '/' + cube_path
        cube2d_dir = data_path + eseq + '/' + cube2d_path
        lidar_dir = data_path + eseq + '/' + lidar_path
        calib_file = data_path + eseq + '/' + calib_path
        f = open(calib_file, 'r')
        lines = f.readlines()
        f.close()
        list_calib = list(map(lambda x: float(x), lines[1].split(',')))
        if len(list_calib) < 4:
            list_calib.append(0.7) # z offset
        frame_dif = int(list_calib[0])
        # List all label files in the directory
        label_files = [f for f in os.listdir(label_dir) if f.startswith("sceneflowgt_") and f.endswith(".npy")]
        label_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

        # List all tesseract files in the directory
        tesseract_files = [f for f in os.listdir(cube_dir) if f.startswith("tesseract_") and f.endswith(".pt")]
        # Sort files by frame number
        tesseract_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        # List all tesseract2d files in the directory
        tesseract_files2d = [f for f in os.listdir(cube2d_dir) if f.startswith("tesseract2d_") and f.endswith(".pt")]
        # Sort files by frame number
        tesseract_files2d.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

        # List all lidar files in the directory
        lidar_files = [f for f in os.listdir(lidar_dir) if f.startswith("os2-64_") and f.endswith(".pcd")]
        lidar_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

        for i, filename in enumerate(label_files):
            # Extract the current frame number
            frame_number = filename.split("_")[1].split(".")[0]
            if frame_number not in frame_dict[eseq]:
                frame_dict[eseq][frame_number] = {}
            else:
                continue
            # Store the current frame path
            # Store the next frame path if it exists
            if i  + frame_dif + 1 < len(tesseract_files):
                frame_dict[eseq][frame_number]["seq"] = eseq
                frame_dict[eseq][frame_number]["frame"] = frame_number
                frame_dict[eseq][frame_number]["tesseract"] = os.path.join(cube_dir, tesseract_files[i + frame_dif])
                frame_dict[eseq][frame_number]["tesseract2d"] = os.path.join(cube2d_dir, tesseract_files2d[i + frame_dif])
                frame_dict[eseq][frame_number]["label"] = os.path.join(label_dir, filename)
                frame_dict[eseq][frame_number]["calib"] = list_calib
                frame_dict[eseq][frame_number]["next_tesseract"] = os.path.join(cube_dir, tesseract_files[i  + frame_dif + 1])
                frame_dict[eseq][frame_number]["next_tesseract2d"] = os.path.join(cube2d_dir, tesseract_files2d[i  + frame_dif + 1])
                frame_dict[eseq][frame_number]["lidar"] = os.path.join(lidar_dir, lidar_files[i])
                
    return frame_dict

# Load Doppler, Range, Azimuth and Elevation Axis
def load_axis(path_head, axis_path, is_in_rad=True, is_with_doppler=True, is_reverse_ae = True):
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

        temp_values = scipy.io.loadmat(path_head + axis_path[0])
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
        if is_reverse_ae:
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
            arr_doppler = scipy.io.loadmat(path_head + axis_path[1])['arr_doppler']
            _, num_3 = arr_doppler.shape
            arr_doppler = arr_doppler.reshape((num_3,))
            return arr_range, arr_azimuth, arr_elevation, arr_doppler
        else:
            return arr_range, arr_azimuth, arr_elevation

def split_dataset(data_dict, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split the dataset into training, validation, and test sets.

    Parameters:
        data_dict (dict): The input dictionary with frame numbers as keys.
        train_ratio (float): Proportion of the dataset for training (default is 0.7).
        val_ratio (float): Proportion of the dataset for validation (default is 0.15).
        test_ratio (float): Proportion of the dataset for testing (default is 0.15).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Three dictionaries (train_set, val_set, test_set) containing the split data.
    """
    # Ensure the ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."
    train_set, val_set, test_set = {}, {}, {}

    # Extract all keys (frame numbers) and shuffle them
    for seq, frame_dict in data_dict.items():

        keys = list(frame_dict.keys())
        random.seed(seed)
        random.shuffle(keys)

        # Calculate split indices
        total_seq_frames = len(keys)
        train_end = int(total_seq_frames * train_ratio)
        val_end = train_end + int(total_seq_frames * val_ratio)

        # Split keys
        train_seq_keys = keys[:train_end]
        val_seq_keys = keys[train_end:val_end]
        test_seq_keys = keys[val_end:]

        # Create sub-dictionaries for each split
        train_set[seq] = {key: frame_dict[key] for key in train_seq_keys}
        val_set[seq] = {key: frame_dict[key] for key in val_seq_keys}
        test_set[seq] = {key: frame_dict[key] for key in test_seq_keys}

    split_set = {
                    'train_set': train_set,
                    'val_set': val_set,
                    'test_set': test_set
                }

    return split_set

# Load Radar Cube Mat
def load_power(cube_path1, cube_path2, is_reverse_ae = True):

    power_path = cube_path1
    power0 = scipy.io.loadmat(power_path)
    power0 = power0["arrDREA"][:,:,:,:]/(1e+13)
    power0 = np.transpose(power0, (0, 1, 3, 2)) #DRAE
    if is_reverse_ae:
        power0 = np.flip(np.flip(power0, axis=2), axis=3)
    power0 = (power0 - np.min(power0)) / (np.max(power0) - np.min(power0))

    next_power_path = cube_path2
    power1 = scipy.io.loadmat(next_power_path)
    power1 = power1["arrDREA"][:,:,:,:]/(1e+13)
    power1 = np.transpose(power1, (0, 1, 3, 2)) #DRAE
    if is_reverse_ae:
        power1 = np.flip(np.flip(power1, axis=2), axis=3)
    power1 = (power1 - np.min(power1)) / (np.max(power1) - np.min(power1))

    cube_tensor0 = torch.tensor(power0).to(dtype=torch.float)
    cube_tensor1 = torch.tensor(power1).to(dtype=torch.float)

    return cube_tensor0, cube_tensor1

def getcoords(params):
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
    coords = coords.to('cuda')
    return coords

def make_lr_lambda(warmup_epochs, decay_epochs, decay_rate):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        else:
            return decay_rate ** ((epoch - warmup_epochs) // decay_epochs)
    return lr_lambda