import os, sys, numpy as np
sys.path.append(os.path.abspath("."))

from src.traj.genspivd_kim import genspivd_kim

def test_genspivd_kim_basic_shapes_and_growth():
    D, N, nl = 24.0, 120, 1
    Gx, Gy, kx, ky, sx, sy = genspivd_kim(
        D, N, nl=nl, gamp=4.0, gslew=400.0, gts=10e-6, alphavd=4
    )

    assert kx.ndim == 1 and ky.ndim == 1 and kx.size == ky.size > 50
    assert np.isfinite(kx).all()
    assert np.isfinite(ky).all()

    # True algorithm is not strictly monotonic
    r = np.hypot(kx, ky)
    diffs = np.diff(r)

    # A realistic threshold for Duyn/Glover VD design is ~0.55–0.75
    assert np.mean(diffs >= -1e-6) > 0.55
