"""
PyTorch-based Non-Uniform Fast Fourier Transform (NUFFT) wrapper for MRI reconstruction.

This module provides a wrapper around torchkbnufft for non-Cartesian MRI reconstruction,
maintaining compatibility with the MATLAB interface while leveraging PyTorch's GPU
acceleration capabilities.

References:
    - torchkbnufft: https://github.com/mmuckley/torchkbnufft
    - Original MATLAB implementation: NUFFT.m, Gdft.m
"""

__version__ = '0.1.0'

try:
    import torchkbnufft as tkbn
except ImportError:
    raise ImportError(
        "torchkbnufft is required for NUFFT operations. "
        "Please install it with: pip install torchkbnufft"
    )

from .nufft_operator import (
    NUFFTOperator,
    KspaceToImage,
    ImageToKspace,
)

# Version compatibility
__torchkbnufft_version__ = tkbn.__version__

__all__ = [
    'NUFFTOperator',
    'KspaceToImage',  # Equivalent to adjoint_nufft in MATLAB
    'ImageToKspace',  # Equivalent to forward_nufft in MATLAB
    '__torchkbnufft_version__',
]
