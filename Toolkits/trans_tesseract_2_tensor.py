import os
import numpy as np
import scipy.io
import sys

import torch
import torch.nn.functional as F


def convert_mat_to_pt(mat_path, save_path, is_reverse_ae=True):
    power = scipy.io.loadmat(mat_path)["arrDREA"] 
    power = np.log10(power)
    power = np.transpose(power, (0, 1, 3, 2))
    if is_reverse_ae:
        power = np.flip(np.flip(power, axis=2), axis=3)
    a_start = (power.shape[2] - 96) // 2
    e_start = (power.shape[3] - 32) // 2
    power = power[:, :128, a_start:a_start+96, e_start:e_start+32]
    power = power.copy()
    tensor = torch.tensor(power, dtype=torch.float32)
    print(tensor.shape)
    tensor = tensor.unsqueeze(0)
    print(tensor.shape)
    tensor = F.avg_pool3d(tensor, kernel_size=(1, 2, 1), stride=(1, 2, 1))
    print(tensor.shape)
    tensor = tensor.squeeze(0)
    print(tensor.shape)
    torch.save(tensor, save_path)
    print(f'trans {mat_path} to {save_path}')

# config settings
seq = ['21', '7', '16', '44', '47','48']   # '1', '5', '30', '35' has trans
for eseq in seq:
    path_data = dict(
        head = '/home/data/K-Radar/K-Radar/generated_files/' + eseq + '/',
        radar_path = 'radar_tesseract',
        tensor_path = 'radar_tensor',)
    load_dir = path_data['head'] + path_data['radar_path']
    save_dir = path_data['head'] + path_data['tensor_path']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"create {save_dir}")
    else:
        print(f"{save_dir} existed")
    for filename in os.listdir(load_dir):
        if filename.endswith(".mat"):
            frame = filename.split('_')[1].split('.')[0]
            file_path = os.path.join(load_dir, filename)
            save_path = os.path.join(save_dir, 'tesseract_'+frame+'.pt')
            convert_mat_to_pt(file_path, save_path)
