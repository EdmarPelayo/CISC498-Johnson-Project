import numpy as np
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from src.traj.genkspace import genkspace
from src.timeseg.time_segmentation import TimeSegmentation


class DummyG:
    def __init__(self, n_data, n_vox):
        self._shape = (n_data, n_vox)

    @property
    def shape(self):
        return self._shape

    def forward(self, x):
        return np.ones(self._shape[0], dtype=np.complex64) * np.sum(x)

    def adjoint(self, y):
        return np.ones(self._shape[1], dtype=np.complex64) * np.sum(y)


def test_time_segmentation_forward_and_adjoint():
    FOV = 24.0
    N = 120
    nint = 4
    tsamp = 5e-6

    kx, ky, gx, gy = genkspace(
        FOV, N,
        ld=0,
        nint=nint,
        gamp=4.0,
        gslew=400.0,
        tsamp=tsamp,
        rotamount=0,
        rev_flag=0,
        gts=tsamp,
        flag_vd=0,
        int_rotation=1,
        alpha_vd=4
    )

    nd = kx.size
    shot_len = nd // nint
    timing = np.arange(shot_len) * tsamp

    field_map = np.zeros((64, 64), dtype=np.float64)
    n_vox = field_map.size

    G = DummyG(n_data=nd, n_vox=n_vox)

    L = 5
    TS = TimeSegmentation(
        G,
        timing,        # <-- FIXED: argument name is tt
        field_map,     # <-- FIXED: argument name is we
        L,
        interpolator="hanning"
    )

    assert TS.shape == (nd, n_vox)
    assert TS.time_interp.shape == (L + 1, timing.size)

    x = np.ones(n_vox, dtype=np.complex64)
    y = TS.forward(x)

    assert y.shape == (nd,)
    assert np.isfinite(y).all()

    xb = TS.adjoint(y)

    assert xb.shape == (n_vox,)
    assert np.isfinite(xb).all()
    assert np.allclose(y, y[0])
