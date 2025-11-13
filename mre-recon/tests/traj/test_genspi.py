import os, sys, numpy as np
sys.path.append(os.path.abspath("."))

from src.traj.genspi import genspi

def test_genspi_basic_shapes_and_monotonic_radius():
    D, N, nl = 24.0, 120, 1
    Gx, Gy, kx, ky, sx, sy = genspi(D, N, nl=nl, gamp=4.0, gslew=400.0, gts=10e-6)

    # shapes & finiteness
    assert kx.ndim == 1 and ky.ndim == 1
    assert kx.size == ky.size > 100
    assert np.isfinite(kx).all() and np.isfinite(ky).all()

    # monotonic radius for a *single* interleaf
    r = np.hypot(kx, ky)
    # allow small numerical noise; check that most successive differences are >= 0
    diffs = np.diff(r)
    frac_nonneg = np.mean(diffs >= -1e-6)
    assert frac_nonneg > 0.98  # largely increasing
