import numpy as np
import torch

def ncol(x):
    """
    Return number of columns in x, emulating MATLAB's size(x, 2).
    
    Handles both NumPy arrays and Torch tensors.
    """
    if isinstance(x, torch.Tensor):
        # Use torch.atleast_2d for torch tensors
        return torch.atleast_2d(x).shape[1]
    
    # Default NumPy behavior
    return np.atleast_2d(x).shape[1]