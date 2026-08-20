import torch
import torch.nn as nn
import lpips
from insightface.app import FaceAnalysis

lpips_alex = lpips.LPIPS(net='alex')
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(160,160))

class IdentityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imgA, imgB):
        raise NotImplementedError('Use ArcFace embedding extraction.')

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return lpips_alex(a, b).mean()
