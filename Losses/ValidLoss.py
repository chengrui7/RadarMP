import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp

import numpy as np

class ValidLoss(nn.Module):
    def __init__(self, params):
        super(ValidLoss, self).__init__()
        self.params = params
        self.pl = PointLoss(params)
        self.fl = FlowLoss(params)
        self.dl = DiceLoss(params)
        
    def forward(self, seg, flow, pcl_label, flow_label):

        # cal metric
        bs, _, ran, azi, ele = pcl_label.shape
        # reduction dop
        seg = seg.permute(0,2,1).contiguous().view(bs, -1, ran, azi, ele)
        flow = flow.permute(0,2,1).contiguous().view(bs, -1, ran, azi, ele)
        dl_ = self.dl(seg, pcl_label)
        plf1_, plpr_, plre_ = self.pl(seg, pcl_label)
        # cal flow loss
        fl_ = self.fl(pcl_label, flow, flow_label)
        loss = dl_
        loss_item = {
            'dice_loss': dl_,
            'seg_losspr': plpr_,
            'seg_lossre': plre_,
            'seg_lossf1': plf1_,
            'flow_loss': fl_,
        }

        return loss, loss_item
    

class DiceLoss(nn.Module):
    def __init__(self, params):
        super(DiceLoss, self).__init__()
        self.params = params
        self.loss_fn = smp.losses.FocalLoss('binary', alpha=0.9, gamma=2.0)
        #pos_weight = torch.tensor([10.0]).to('cuda')
        #self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, seg, pcllabel):
        # segloss = self.loss_fn(seg, pcllabel)
        # segloss = segloss.sum() / pcllabel.sum()
        pcl = torch.sigmoid(seg)
        intersection = (pcl * pcllabel).sum()
        diceloss = 1 - (2 * intersection + 1e-5) / (pcl.sum() + pcllabel.sum() + 1e-5)
        return diceloss

class PointLoss(nn.Module):
    def __init__(self, params):
        super(PointLoss, self).__init__()
        self.params = params

    def forward(self, pcl, label_pcl):
        pcl = torch.sigmoid(pcl)
        # plot
        preds = pcl.detach().cpu().numpy()
        plt.hist(preds.ravel(), bins=100)
        plt.title("Histogram of Sigmoid Outputs")
        plt.xlabel("Probability")
        plt.ylabel("Frequency")
        plt.savefig("sigmoid_histogram_val.png") 
        plt.close()
        pcl = (pcl > 0.5).float()
        
        # True Positives (TP)
        tp = (pcl * label_pcl).sum()
        # False Positives (FP)
        fp = (pcl * (1 - label_pcl)).sum()
        # False Negatives (FN)
        fn = ((1 - pcl) * label_pcl).sum()
        
        # Precision and Recall
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        
        # F1 Score 
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        return f1, precision, recall
    
class FlowLoss(nn.Module):
    def __init__(self, params):
        super(FlowLoss, self).__init__()
        self.params = params
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

    def forward(self, label_pcl, flow, label_flow):

        # label preprocess
        flow = label_pcl * flow
        label_flow = label_flow * label_pcl

        loss = self.loss_fn(flow, label_flow)
        loss = loss.sum() / (label_pcl.sum() + 1e-6)

        return loss