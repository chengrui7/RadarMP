import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from Utils import IOStream

class LossRecord:

    def __init__(self, params, num_batch, num_val_batch, num_epoch, log_dir, ios=None):
        self.params = params
        self.reset_epoch()
        self.num_batch = num_batch
        self.num_val_batch = num_val_batch
        self.num_epoch = num_epoch 
        self.ios = ios
        self.train_loss_epoch_all = []  # List of dicts
        self.val_loss_epoch_all = []    # List of dicts
        self.gt_loss_epoch_all = []
        self.writer = SummaryWriter(log_dir=log_dir)

    def reset_epoch(self):
        self.train_loss_epoch = []  # List of dicts
        self.val_loss_epoch = []
        self.gt_loss_epoch = []

    def record_batch_loss(self, loss_dict: dict, weight=None, mode='train', batch_idx=None, epoch_idx=None):
        """
        :param loss_dict: dict, e.g., {"loss_total": 0.51, "loss_flow": 0.32}
        :param mode: 'train' or 'val'
        :param batch_idx: optional, for printing batch index
        """
        assert mode in ['train', 'val', 'gt']
        loss_dict = {k: float(v) for k, v in loss_dict.items()}  # Ensure float
        num_batch = self.num_batch
        if mode == 'train':
            self.train_loss_epoch.append(loss_dict)
        elif mode == 'gt':
            self.gt_loss_epoch.append(loss_dict)
        else:
            self.val_loss_epoch.append(loss_dict)
            num_batch = self.num_val_batch

        # Print batch loss
        if self.ios is not None and weight is not None:
            loss_str = ', '.join([f'{k}: {v:.4f}' for k, v in loss_dict.items()])
            alpha_str = ', '.join([f'w{i}: {a:.8f}' for i, a in enumerate(weight)])
            prefix0 = f"Epoch {epoch_idx}/{self.num_epoch}" if epoch_idx is not None else f"Epoch"
            prefix = f"[{mode.upper()}] Batch {batch_idx}/{num_batch}" if batch_idx is not None else f"[{mode.upper()}] Batch"
            self.ios.cprint(f"{prefix0} - {prefix} : {loss_str} || {alpha_str}")
        elif self.ios is not None and weight is None:
            loss_str = ', '.join([f'{k}: {v:.4f}' for k, v in loss_dict.items()])
            prefix0 = f"Epoch {epoch_idx}/{self.num_epoch}" if epoch_idx is not None else f"Epoch"
            prefix = f"[{mode.upper()}] Batch {batch_idx}/{num_batch}" if batch_idx is not None else f"[{mode.upper()}] Batch"
            self.ios.cprint(f"{prefix0} - {prefix} : {loss_str}")

    def compute_avg_loss(self, mode='train'):
        """
        Compute average of each loss key across the epoch.
        :return: dict of avg losses
        """
        if mode == 'train':
            loss_list = self.train_loss_epoch
        elif mode == 'gt':
            loss_list = self.gt_loss_epoch
        else:
            loss_list = self.val_loss_epoch

        if not loss_list:
            return {}

        # Sum over all batch dicts
        keys = loss_list[0].keys()
        avg_loss = {}
        for k in keys:
            avg_loss[k] = sum(d[k] for d in loss_list) / len(loss_list)
        return avg_loss

    def record_epoch_end(self, mode='train', epoch_idx=None):
        """
        Call at the end of an epoch to compute & print average loss
        """
        avg_loss = self.compute_avg_loss(mode)
        if self.ios != None:
            loss_str = ', '.join([f'{k}: {v:.4f}' for k, v in avg_loss.items()])
            epoch_info = f"Epoch {epoch_idx}" if epoch_idx is not None else "Epoch"
            self.ios.cprint(f"[{mode.upper()}] {epoch_info} - Avg Losses: {loss_str}")


        if epoch_idx is not None:
            for k, v in avg_loss.items():
                self.writer.add_scalar(f"{mode}/{k}", v, epoch_idx)

        # Store epoch average
        if mode == 'train':
            self.train_loss_epoch_all.append(avg_loss)
        elif mode == 'gt':
            self.gt_loss_epoch_all.append(avg_loss)
        else:
            self.val_loss_epoch_all.append(avg_loss)

        # Reset per-epoch loss
        self.reset_epoch()

        return avg_loss

