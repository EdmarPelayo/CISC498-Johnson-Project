# CISC498-Johnson-Project
# MRE Non-Cartesian MRI Reconstruction (MATLAB → PyTorch Conversion)

This project aims to convert the Johnson Lab’s non-Cartesian MRI reconstruction pipeline from MATLAB into a modern, modular, and GPU-accelerated Python/PyTorch framework.

## Background
Magnetic Resonance Elastography (MRE) measures the mechanical stiffness of tissue.  
The Johnson Lab develops high-resolution **spiral k-space** imaging methods for brain imaging applications (e.g., Alzheimer’s research).  
However, current MATLAB implementations are difficult to maintain and slow to run, especially for iterative reconstruction and multi-coil processing.

> The goal of this project is to implement a clean, open Python/PyTorch pipeline for NUFFT-based non-Cartesian MRI reconstruction to improve usability, speed, and research flexibility.  
> :contentReference[oaicite:0]{index=0}

## Objectives
- Replace MATLAB code with a fully **PyTorch-based** implementation
- Support **GPU acceleration** (NVIDIA CUDA + Apple MPS + CPU fallback)
- Use **NUFFT operators** (not gridding) for forward/adjoint image models
- Implement **iterative reconstruction solvers**:
  - Conjugate Gradient (CG)
  - PWLS / Tikhonov / Total Variation regularizers
  - Optionally: FISTA / ADMM / Learned Unrolled Networks
- Build a **modular pipeline** allowing rapid research modifications
- Provide **Jupyter/Colab demos** for reproducibility

## Key Components to Port from MATLAB
| MATLAB Module | Python Equivalent (to build or adopt) |
|---------------|---------------------------------------|
| `NUFFT.m`, `Gdft.m` | Use `torchkbnufft` or custom PyTorch NUFFT ops |
| `SENSE.m` (coil combination) | Torch-based sensitivity multiplication w/ broadcasting |
| `TimeSegmentation.m`, `int_tim_seg.m` | Port to PyTorch (phase correction kernels) |
| `solve_pwls_pcg.m` | Rewrite using Torch autograd + custom linear operator classes |
| `C_sparse.m`, `diag_sp.m`, `Robj.m` | Implement modular regularizers in PyTorch |
| Spiral trajectory generation (`genkspace.m`, `genspi*`) | Port trajectory and sampling operator code |

## Repository Layout (Recommended)


## Dependencies
- Python 3.9+
- PyTorch (CUDA or MPS recommended)
- `torchkbnufft` or `tfkbnufft` (we will evaluate performance)
- NumPy, SciPy, Matplotlib
- JupyterLab / Google Colab

## Getting Started (Colab Friendly)
```bash
git clone https://github.com/<YOUR_USERNAME>/mre-recon.git
cd mre-recon
pip install -r requirements.txt
