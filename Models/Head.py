import torch
import torch.nn as nn

class FlowHead(nn.Module):
    def __init__(self, input_channels=64):
        super(FlowHead, self).__init__()
        self.feature_proj = nn.Sequential(nn.Linear(input_channels, input_channels//2),
                                            nn.ReLU(),
                                            nn.LayerNorm(input_channels//2),
                                            nn.Linear(input_channels//2, input_channels//4),
                                            nn.ReLU(),)
        self.flow_head = nn.Linear(in_features=input_channels//4, out_features=3)
    def forward(self, x1):
        x1 = self.feature_proj(x1)
        flow = self.flow_head(x1)
        return flow

class SegFlowHead(nn.Module):
    def __init__(self, seg_channels=64, corr_channels=128, dropout=0.1):
        super(SegFlowHead, self).__init__()
        self.seg_proj1 = nn.Sequential(nn.Linear(seg_channels, 128),
                                        nn.ReLU(),
                                        nn.LayerNorm(128),)
        self.seg_proj2 = nn.Sequential(nn.Linear(128*2, 64),
                                        nn.ReLU(),
                                        nn.Linear(in_features=64, out_features=32),
                                        nn.GELU(),
                                        nn.Linear(in_features=32, out_features=1))
        self.corr_proj1 = nn.Sequential(nn.Linear(corr_channels, 128),
                                        nn.ReLU(),
                                        nn.LayerNorm(128),)
        self.corr_proj2 = nn.Sequential(nn.Linear(128*2, 64),
                                        nn.ReLU(),
                                        nn.Linear(in_features=64, out_features=32),
                                        nn.GELU(),
                                        nn.Linear(in_features=32, out_features=3))

    def forward(self, x1, x1x2):

        seg = self.seg_proj1(x1)
        flow = self.corr_proj1(x1x2)
        
        segflow = torch.cat([seg, flow], dim=-1)

        seg = self.seg_proj2(segflow)
        flow = self.corr_proj2(segflow)
        return seg, flow

class SegHead(nn.Module):
    def __init__(self, ksize=500, temperature=1.0):
        super(SegHead, self).__init__()
        self.ksize = ksize
    def forward(self, x):
        B, D, R, A, E = x.shape
        x_mean = x.mean(dim=1, keepdim=True)  # (B, 1, R, A, E)
        x_flat = x_mean.view(B, -1)           # (B, R*A*E)

        # topk
        topk_values, topk_indices = torch.topk(x_flat, k=self.ksize, dim=1)      # (B, k)
        mask = torch.zeros_like(x_flat)
        mask.scatter_(dim=1, index=topk_indices, src=10*topk_values)  # (B, 1, R, A, E)
        mask = mask.unsqueeze(-1)                                      
        return mask
        