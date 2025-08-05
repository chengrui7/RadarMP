import os
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from multiprocessing import Pool, cpu_count

def convert_mat_to_pt(mat_path, save_path, is_reverse_ae=True):
    power = scipy.io.loadmat(mat_path)["arrDREA"]
    power = np.log10(power)
    power = np.transpose(power, (0, 1, 3, 2))  # (D, R, A, E)

    if is_reverse_ae:
        power = np.flip(power, axis=2)
        power = np.flip(power, axis=3)

    a_start = (power.shape[2] - 96) // 2
    e_start = (power.shape[3] - 32) // 2
    power = power[:, :128, a_start:a_start+96, e_start:e_start+32]  # (D=64, R/2=128, A=96, E=32)
    power = power.copy()
    
    tensor = torch.tensor(power, dtype=torch.float32)
    tensor = tensor.unsqueeze(0)  # (1, D, R, A, E)
    tensor = F.avg_pool3d(tensor, kernel_size=(1, 2, 1), stride=(1, 2, 1))  # (1, D, R/2, A/2, E)
    tensor = tensor.squeeze(0)
    tensor_ra = tensor.mean(dim=(0, 3)) 
    tensor_re = tensor.mean(dim=(0, 2))
    tensor_ae = tensor.mean(dim=(0, 1))
    torch.save({
    'tensor_ra': tensor_ra,
    'tensor_re': tensor_re,
    'tensor_ae': tensor_ae,
    }, save_path)
    print(f'[✓] Saved: {save_path}')

def process_sequence_parallel(eseq, num_workers=8):
    head = f'/home/data/K-Radar/K-Radar/generated_files/{eseq}/'
    load_dir = os.path.join(head, 'radar_tesseract')
    save_dir = os.path.join(head, 'radar_tensor2d')
    os.makedirs(save_dir, exist_ok=True)

    mat_files = sorted([f for f in os.listdir(load_dir) if f.endswith(".mat")])
    tasks = []

    for filename in mat_files:
        frame = filename.split('_')[1].split('.')[0]
        file_path = os.path.join(load_dir, filename)
        save_path = os.path.join(save_dir, f'tesseract2d_{frame}.pt')
        tasks.append((file_path, save_path))

    with Pool(processes=num_workers) as pool:
        pool.starmap(convert_mat_to_pt, tasks)

if __name__ == "__main__":
    seq_list = ['7', '16', '44', '42'] # '1','5','30','35','21',
    for seq in seq_list:
        process_sequence_parallel(seq, num_workers=min(4, cpu_count()))