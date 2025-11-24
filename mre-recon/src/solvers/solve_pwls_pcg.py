# src/solvers/solve_pcg.py
from __future__ import annotations
import torch
from typing import Callable, Optional, Union

from ..regularizers.robj import Robj

@torch.no_grad()
def solve_pwls_pcg(
    x0: torch.Tensor,                       # [N] or [H*W] (real or complex)
    A,                                      # has .forward(x)->y and .adjoint(y)->x
    w: Union[float, torch.Tensor],          # scalar or per-sample weights
    y: torch.Tensor,                        # [Nsamp] or any shape A.forward(x) returns
    R: Optional[Robj],                      # Robj instance with .cgrad(x); can be None for no reg
    niter: int = 10,
    M: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,  # preconditioner
    tol: float = 0.0,
) -> torch.Tensor:
    """
    Penalized Weighted Least Squares via PCG.

    Minimizes:  || A x - y ||_W^2 + R(x)

    where:
      - A.forward(x)  -> y_pred
      - A.adjoint(r)  -> backprojected residual
      - R.cgrad(x)    -> gradient of regularizer from Robj

    Solves the normal equations:
        (A^H W A + ∇R) x = A^H W y
    
    """

    # Work with a flat vector internally
    x = x0.reshape(-1)

    # Helper to apply data weights W
    def W(z: torch.Tensor) -> torch.Tensor:
        if isinstance(w, (int, float, complex)):
            return z * w
        else:
            # assume w is a tensor broadcastable to z
            return z * w

    def normal_op(v: torch.Tensor) -> torch.Tensor:
        """
        Apply the normal-equation operator:
            v ↦ A^H W A v + ∇R(v)
        """
        v_flat = v.reshape(-1)

        # Forward model: A v
        Av = A.forward(v_flat)

        # Apply data weights
        WAv = W(Av)

        # Adjoint: A^H (W A v)
        AtWAv = A.adjoint(WAv).reshape(-1)

        # If no regularizer was provided, just return data term
        if R is None:
            return AtWAv

        # Regularizer gradient from Robj: ∇R(x)
        # R.cgrad expects a vector (same layout as x)
        Rv = R.cgrad(v_flat)

        # Total normal-equation operator: A^H W A v + ∇R(v)
        return AtWAv + Rv

    # Right-hand side: b = A^H W y
    b = A.adjoint(W(y)).reshape(-1)

    # Initial residual and direction
    xk = x.clone()
    rk = b - normal_op(xk)
    zk = M(rk) if M is not None else rk
    pk = zk.clone()
    rz_old = (rk.conj() * zk).real.sum()

    for _ in range(niter):
        Apk = normal_op(pk)
        denom = (pk.conj() * Apk).real.sum()
        # Avoid divide by zero if something degenerates
        if denom == 0:
            break

        alpha = rz_old / denom

        xk = xk + alpha * pk
        rk = rk - alpha * Apk

        # Stopping criterion
        if tol > 0 and rk.norm() < tol:
            break

        zk = M(rk) if M is not None else rk
        rz_new = (rk.conj() * zk).real.sum()
        if rz_old == 0:
            break
        beta = rz_new / rz_old

        pk = zk + beta * pk
        rz_old = rz_new

    # Return in same shape as input x0
    return xk.view_as(x0)
