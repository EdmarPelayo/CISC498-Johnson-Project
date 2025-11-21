import torch
import numpy as np
from typing import Optional, List, Union
from scipy import sparse
# import sys
from .c_sparse import c_sparse
from ..utils.ncol import ncol

class Robj:
    """
    Build roughness penalty regularization "object" for regularized solutions 
    to inverse problems. (Python/Torch version of MATLAB Robj)
    """
    
    def __init__(
        self,
        mask: torch.Tensor,
        beta: float = 1.0,
        delta: float = 0.001,
        potential: str = 'quad',
        dims2penalize: Optional[List[int]] = None
    ):
        # Validate inputs
        if mask.dtype != torch.bool:
            raise ValueError('Mask must be logical (boolean)')
        
        if not (isinstance(beta, (int, float)) and beta >= 0):
            raise ValueError('Beta must be numeric scalar >= 0')
        
        if not (isinstance(delta, (int, float)) and delta >= 0):
            raise ValueError('Delta must be numeric scalar >= 0')
        
        # Store parameters
        self.beta = beta
        self.potential = potential
        self.delta = delta
        self.mask = mask
        self.dtype = torch.float32 # Use float32 for consistency
        
        # Pass None to c_sparse to use its default (all dims)
        self.dims2penalize = dims2penalize
        
        # Initialize offsets (kept for compatibility, though not used by c_sparse.py)
        if mask.ndim == 2:
            nx, _ = mask.shape
            self.offsets = [1, nx, nx + 1, nx - 1]
        elif mask.ndim == 3:
            nx, ny, _ = mask.shape
            self.offsets = [1, nx, nx * ny]
        else:
            raise ValueError('Only 2D and 3D support')
        self.offsets = [int(off) for off in self.offsets]
        
        # --- CRITICAL TYPE CONVERSION ---
        # 1. Convert torch mask to numpy for c_sparse
        mask_np = mask.cpu().numpy().astype(np.uint8)
        
        # 2. Call c_sparse (returns scipy.sparse.csr_matrix and np.ndarray)
        C1_scipy, wt_np = c_sparse(mask_np, self.dims2penalize)
        
        # 3. Convert SciPy sparse matrix to Torch sparse tensor
        C1_coo = C1_scipy.tocoo()
        C1_indices = torch.tensor(
            np.vstack((C1_coo.row, C1_coo.col)), 
            dtype=torch.long
        )
        C1_values = torch.tensor(C1_coo.data, dtype=self.dtype)
        self.C1 = torch.sparse_coo_tensor(
            C1_indices, C1_values, C1_coo.shape
        ).coalesce()
        
        # 4. Convert weights to a Torch tensor
        self.wt = torch.tensor(
            self.beta * wt_np.flatten(), 
            dtype=self.dtype
        ).squeeze()
        # --- END OF TYPE CONVERSION ---
        
        # Setup potential functions
        if self.potential == 'quad':
            self.potk = lambda t: (torch.abs(t) ** 2) / 2
            self.wpot = lambda t: torch.ones_like(t)
            self.dpot = lambda t: t
        elif self.potential == 'approxTV':
            delta_val = self.delta
            # Note: Correcting MATLAB bug (using 't' not 'd')
            self.potk = lambda t: (delta_val ** 2) * (
                torch.sqrt(1.0 + (torch.abs(t) / delta_val) ** 2) - 1.0
            )
            self.wpot = lambda t: 1.0 / torch.sqrt(
                1.0 + (torch.abs(t) / delta_val) ** 2
            )
            self.dpot = lambda t: t / torch.sqrt(
                1.0 + (torch.abs(t) / delta_val) ** 2
            )
        else:
            raise ValueError(f"Unknown potential: {self.potential}")
    
    def cgrad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate gradient of penalty function.
        """
        # Ensure x is the correct dtype and shape
        x_vec = x.to(self.dtype).squeeze()
        if x_vec.ndim == 1:
            x_vec = x_vec.unsqueeze(-1)
        
        # C1 * x
        Cx = torch.sparse.mm(self.C1, x_vec)
        # dpot(C1 * x)
        dCx = self.dpot(Cx)
        
        # This is the translation of `diag_sp(wt) * dCx`
        # It's just element-wise multiplication
        weighted = self.wt.unsqueeze(-1) * dCx
        
        # C1' * (weighted vector)
        out = torch.sparse.mm(self.C1.t(), weighted)
        return out.squeeze()
    
    def penal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate penalty function R(x).
        """
        # Ensure x is the correct dtype and shape
        x_vec = x.to(self.dtype).squeeze()
        if x_vec.ndim == 1:
            x_vec = x_vec.unsqueeze(-1)
            
        # C1 * x
        Cx = torch.sparse.mm(self.C1, x_vec)
        # potk(C1 * x)
        pot_values = self.potk(Cx)
        
        # Replicate wt for number of columns in x
        n_cols = ncol(x_vec) # ncol() is now torch-aware
        wt_rep = self.wt.unsqueeze(-1).repeat(1, n_cols)
        
        # sum(wt .* potk(C1 * x))
        out = torch.sum(wt_rep * pot_values)
        return out
    
    def denom(self, ddir: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate denominator for separable surrogate.
        """
        # Ensure correct dtypes and shapes
        x_vec = x.to(self.dtype).squeeze().unsqueeze(-1)
        ddir_vec = ddir.to(self.dtype).squeeze().unsqueeze(-1)

        # C1 * ddir
        Cdir_full = torch.sparse.mm(self.C1, ddir_vec)
        # C1 * x
        Cx_full = torch.sparse.mm(self.C1, x_vec)
        
        # Create mask from weights
        mask = (self.wt > 0).unsqueeze(-1)
        
        Cdir = Cdir_full * mask
        Cx = Cx_full * mask
        
        # wpot(Cx)
        wpot_Cx = self.wpot(Cx)
        # wpot(Cx) .* Cdir
        weighted = wpot_Cx * Cdir # Element-wise
        
        # Cdir' * (weighted vector)
        out = (Cdir.t() @ weighted)
        return out.squeeze()