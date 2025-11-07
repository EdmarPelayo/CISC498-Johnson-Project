# src/sense/sense_operator.py
from __future__ import annotations
import torch

class SenseOp:
    """
    Wrap A with coil sensitivities.
    Mirrors: G = SENSE(G, senseMap_reshaped)
    MATLAB passes [N*N, nCoils]; here use [coils, H, W].
    """
    def __init__(self, A, smaps: torch.Tensor):
        """
        smaps: [coils, H, W] complex64, unit-norm preferred
        """
        self.A = A
        self.smaps = smaps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [H*W] complex
        H, W = self.smaps.shape[-2:]
        x_img = x.view(H, W)
        y_list = []
        for c in range(self.smaps.shape[0]):
            y_list.append(self.A.forward((self.smaps[c] * x_img).reshape(-1)))
        return torch.stack(y_list, dim=0)  # [coils, Nsamp]

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        # y: [coils, Nsamp]
        H, W = self.smaps.shape[-2:]
        acc = torch.zeros(H*W, dtype=y.dtype, device=y.device)
        for c in range(self.smaps.shape[0]):
            img_c = self.A.adjoint(y[c]).view(H, W)
            acc += (self.smaps[c].conj() * img_c).reshape(-1)
        return acc
