# src/solvers/recon.py
from __future__ import annotations
import torch
from typing import Optional
from ..solvers.solve_pcg import solve_pwls_pcg
from ..timeseg.time_segmentation import TimeSegmentedOp
from ..sense.sense_operator import SenseOp

def nufft_recon(
    xinit: torch.Tensor,       # [H,W] complex64
    G,                         # NufftOp (for this slice)
    rawData: torch.Tensor,     # [Nsamp] or [coils,Nsamp] complex64
    R,                         # QuadR or similar
    *,
    fieldCorrection: bool=False,
    timingVec: Optional[torch.Tensor]=None,  # [Nsamp]
    fieldMap: Optional[torch.Tensor]=None,   # [H,W]
    L: int=20,
    senseCorrection: bool=False,
    senseMap: Optional[torch.Tensor]=None,   # [coils,H,W]
    niter: int=10,
):
    A = G
    if fieldCorrection:
        assert timingVec is not None and fieldMap is not None
        A = TimeSegmentedOp(A, timingVec, fieldMap, L)
    if senseCorrection:
        assert senseMap is not None
        A = SenseOp(A, senseMap)

    x0 = xinit.reshape(-1)
    # MATLAB packs slice-by-slice; here rawData is already per-slice
    y = rawData if rawData.ndim == 1 else rawData  # [Nsamp] or [coils,Nsamp]
    x_hat = solve_pwls_pcg(x0, A, w=1.0, y=y.reshape(-1) if y.ndim==1 else y, R=R, niter=niter)
    return x_hat.view_as(xinit)
