import torch
import torch.nn as nn

class SegAttention(nn.Module):
    def __init__(self, RAE64_channels, AE_channels, output_channels, rae64_d_model=64, ae_d_model=64, num_heads=4, num_layers=4):
        super(SegAttention, self).__init__()  
        # seg RAE64 attention
        self.project_featRAE64 = nn.Linear(RAE64_channels, rae64_d_model)
        self.project_coorRAE64 = nn.Linear(3*4*4*4, rae64_d_model)
        encoder_layerRAE64 = nn.TransformerEncoderLayer(d_model=rae64_d_model, nhead=num_heads, batch_first=True)
        self.RAE64_selfattention = nn.TransformerEncoder(encoder_layerRAE64, num_layers=num_layers)
        # seg AE attention
        self.project_featAE = nn.Linear(AE_channels, ae_d_model)
        self.project_coordAE = nn.Linear(3, ae_d_model)
        encoder_layerAE = nn.TransformerEncoderLayer(d_model=ae_d_model, nhead=num_heads, batch_first=True)
        self.AE_selfattention = nn.TransformerEncoder(encoder_layerAE, num_layers=num_layers)
        # seg out
        self.output_proj = nn.Linear(rae64_d_model//64 + 1, output_channels)
    
    def forward(self, x, oriRAE64, oriAE):
        # init
        device = x.device
        bs, cx, ran, azi, ele = x.shape
        # RAE 4*4*4 seg
        x_RAE64 = x.view(bs, cx, ran//4, 4, azi//4, 4, ele//4, 4).permute(0, 2, 4, 6, 3, 5, 7, 1)
        x_RAE64 = x_RAE64.contiguous().view(bs, (ran//4*azi//4*ele//4), 4*4*4*cx)
        x_RAE64 = self.project_featRAE64(x_RAE64)
        coord_RAE64 = self.project_coorRAE64(oriRAE64)
        x_RAE64 = x_RAE64 + coord_RAE64
        x_RAE64 = self.RAE64_selfattention(x_RAE64)
        bs, num64, cx64 = x_RAE64.shape
        x_RAE64 = x_RAE64.permute(0,2,1).contiguous().view(bs, cx64, ran//4, azi//4, ele//4)
        x_RAE64 = x_RAE64.view(bs, cx64//64, 4, 4, 4, ran//4, azi//4, ele//4)  # [bs, cx, 4, 4, 4, ran//4, azi//4, ele//4]
        x_RAE64 = x_RAE64.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()  # [bs, cx, ran//4, 4, azi//4, 4, ele//4, 4]
        x_RAE64 = x_RAE64.view(bs, cx64//64, ran, azi, ele)
        # AE seg
        x_AE = x.permute(0, 3, 4, 2, 1).contiguous().view(bs, azi*ele, ran*cx)
        x_AE = self.project_featAE(x_AE)
        coord_AE = self.project_coordAE(oriAE)
        x_AE = x_AE + coord_AE
        x_AE = self.AE_selfattention(x_AE)
        x_AE = x_AE.view(bs, azi, ele, ran).permute(0, 3, 1, 2)  # (bs, ran, azi, ele)
        x_AE = x_AE.unsqueeze(1)               # (bs, 1, ran, azi, ele)
        # fuse
        x_fused = torch.cat([x_RAE64, x_AE],dim=1)
        x_fused = x_fused.permute(0,2,3,4,1).contiguous().view(bs, ran*azi*ele, -1)
        x_fused = self.output_proj(x_fused)
        return x_fused