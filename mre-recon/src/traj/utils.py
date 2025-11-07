import numpy as np
import torch

def normalize_ktraj(kx, ky, N, FOV, backend="kbnufft", device="cpu"):
    """
    Convert (kx, ky) from pixel-index units to normalized NUFFT coords.

    Parameters
    ----------
    kx, ky : np.ndarray, shape (M,)
        k-space coordinates in original index units.
    N : int
        Matrix size (assumes square for now).
    FOV : float
        Field of view in cm (unused for now but kept for consistency).
    backend : {"kbnufft", "sigpy"}
        Controls output scaling.
    """
    k = np.stack([kx, ky], axis=0)  # shape (2, M)
    
    if backend == "kbnufft":
        # Scale to [-π, π] range
        k = k * (2 * np.pi / N)
    elif backend == "sigpy":
        # Scale to [-0.5, 0.5] range
        k = k / N
    else:
        raise ValueError("backend must be 'kbnufft' or 'sigpy'.")

    return torch.tensor(k, dtype=torch.float32, device=device)
