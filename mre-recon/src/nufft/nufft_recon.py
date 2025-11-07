import torch
import torch.nn as nn
from torchkbnufft import KbNufft, KbNufftAdjoint

# SENSE NUFFT operator : Wrapper to apply coil sensitivities

class NUFFTOp(nn.Module):
    def __init__(self, im_size, grid_size=None, numpoints=6):
        super().__init__()
        self.kb_nufft = KbNufft(im_size=im_size, grid_size=self.grid_size, numpoints=numpoints)
        self.kb_nufft_adj = KbNufftAdjoint(im_size=im_size, grid_size=self.grid_size, numpoints=numpoints)
        
    def forward(self, image, ktraj, dcf=None):
        '''
        image: complex tensor (B,H,W) or (B,H,W,D)
        ktraj: normalized k-space coordinates (B,2,M) for 2D or (B,3,M) for 3D
        returns: complex k-space samples (B,M) or (B,M,D)
        '''
        kdata = self.kb_nufft(image, ktraj)
        if dcf is not None:
            kdata = kdata  * dcf 
        return kdata
        
    def adjoint(self, samples, ktraj, dcf = None):
        '''
        samples: complex k-space samples (B,M) or (B,M,D)
        ktraj: normalized k-space coordinates (B,2,M) for 2D or (B,3,M) for 3D
        returns: complex image tensor (B,H,W) or (B,H,W,D)
        non-uniform samples => gridded image (complex)
        '''
        if dcf is not None:
            samples = samples * dcf
        img = self.kb_nufft_adj(samples, ktraj)
        return img
        