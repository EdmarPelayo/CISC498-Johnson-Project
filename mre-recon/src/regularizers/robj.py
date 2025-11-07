# src/regularizers/robj.py
import torch

class QuadR:
    def __init__(self, beta: float):
        self.beta = float(beta)
    def grad(self, x: torch.Tensor) -> torch.Tensor:
        # ∂/∂x (beta * ||x||^2) = 2*beta*x
        return (2.0 * self.beta) * x
