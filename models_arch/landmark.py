import os
import torch
import torch.nn as nn
from einops import rearrange
from .mobilefacenet import MobileFaceNet

class LandmarkDetector(nn.Module):
    def __init__(self, weights_path=None, map_location='cpu'):
        super().__init__()
        self.size = 112
        self.backbone = MobileFaceNet([112, 112], 136)
        
        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=map_location, weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict)
            
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        if x.shape[2] != self.size or x.shape[3] != self.size:
            x = nn.functional.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=True)
        _, feat = self.backbone(x)
        return feat
