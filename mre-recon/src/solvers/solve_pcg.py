# src/solvers/solve_pcg.py
from __future__ import annotations
import torch
from typing import Callable, Optional

@torch.no_grad()
def solve_pwls_pcg(
    x0: torch.Tensor,                # [H*W] complex64
    A,                               # has .forward(x)->y and .adjoint(y)->x
    w: float,                        # data weight (MATLAB uses 1)
    y: torch.Tensor,                 # [Nsamp] or [coils,Nsamp] complex64
    R,                               # has .grad(x): same shape as x
    niter: int = 10,
    M: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,  # precond
    tol: float = 0.0,
) -> torch.Tensor:
    """
    PCG on normal equations: (A^H W A + λR) x = A^H W y
    """
    device = x0.device
    x = x0.clone()

    def normal_op(v: torch.Tensor) -> torch.Tensor:
        Av = A.forward(v)
        AtAv = A.adjoint(w * Av)
        return AtAv + R.grad(v)

    b = A.adjoint(w * y)

    r = b - normal_op(x)
    z = M(r) if M is not None else r
    p = z.clone()
    rz_old = (r.conj() * z).real.sum()

    for _ in range(niter):
        Ap = normal_op(p)
        alpha = rz_old / (p.conj() * Ap).real.sum()
        x = x + alpha * p
        r = r - alpha * Ap
        if tol > 0 and r.norm() < tol:
            break
        z = M(r) if M is not None else r
        rz_new = (r.conj() * z).real.sum()
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    return x
