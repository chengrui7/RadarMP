import torch
import torch.nn as nn
import torch.nn.functional as F

class DopplerMLP(nn.Module):
    def __init__(self, input_channels, coord_channels, output_channels, dropout=0.2):
        super(DopplerMLP, self).__init__()  

        self.feat_encoder = nn.Sequential(nn.Linear(input_channels, coord_channels),
                                            nn.ReLU(),
                                            nn.LayerNorm(coord_channels),
                                            nn.Linear(coord_channels, coord_channels),
                                            nn.ReLU(),
                                            nn.Dropout(dropout),)
        self.feat_reduction = nn.Sequential(nn.Linear(coord_channels, output_channels-2),
                                            nn.ReLU(),
                                            nn.LayerNorm(output_channels-2),
                                            nn.Linear(output_channels-2, output_channels-2),
                                            nn.ReLU(),
                                            nn.Dropout(dropout),)
    
    def forward(self, x, coord):
        device = coord.device
        coord = coord.unsqueeze(1)
        softmax_x = F.softmax(x, dim=-1)
        max_x = F.gumbel_softmax(x, tau=1.0, hard=True, dim=-1)
        dop1 = max_x * coord
        dop1 = torch.sum(dop1, dim=-1, keepdim=True)
        dop2 = softmax_x * coord
        dop2 = torch.sum(dop2, dim=-1, keepdim=True)
        x = self.feat_encoder(x)
        x = x + coord
        x = self.feat_reduction(x)

        x = torch.cat([x, dop1, dop2], dim=-1)

        return x