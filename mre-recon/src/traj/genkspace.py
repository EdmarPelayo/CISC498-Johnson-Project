# genkspace.py
# NumPy port of genkspace.m (wrapper that calls genspi / genspivd_Kim and resamples)
import numpy as np
from .genspi import genspi
from .genspivd_kim import genspivd_kim

def genkspace(FOV, N, ld, nint, gamp, gslew, tsamp,
              rotamount=0, rev_flag=0, gts=4e-6, flag_vd=0, int_rotation=1, alpha_vd=4):
    """
    Parameters mirror MATLAB genkspace.m.
    Returns
    -------
    kx, ky : (nk*nint,) flattened interleaved spiral samples at tsamp spacing
    gx, gy : (ng*nint,) gradients (optional; None if not needed)
    """
    # choose base generator
    if flag_vd == 1:
        Gx, Gy, kxi, kyi, _, _ = genspivd_kim(FOV, N, nint, gamp, gslew, gts, alpha_vd)
    else:
        Gx, Gy, kxi, kyi, _, _ = genspi(FOV, N, nint, gamp, gslew, gts)

    # original time grids at generator raster gts
    t_gen = np.arange(0, len(kxi) * gts, gts)[:len(kxi)]
    t_tar = np.arange(0, len(t_gen) * gts, tsamp)
    # linear resample like interp1
    kxt = np.interp(t_tar, t_gen, kxi)
    kyt = np.interp(t_tar, t_gen, kyi)

    # nk from ld / nint (if 0 -> auto from kxt length - 2)
    nk = int(round(ld / nint)) if ld != 0 else 0
    if nk == 0:
        nk = max(len(kxt) - 2, 1)

    # take first nk
    kxo = kxt[:nk].copy()
    kyo = kyt[:nk].copy()

    # rotation amount in quarter-turns as MATLAB: phir = -rotamount * pi/2
    phir = -rotamount * np.pi / 2.0
    kxop = kxo * np.cos(phir) - kyo * np.sin(phir)
    kyop = kyo * np.cos(phir) + kxo * np.sin(phir)

    if rev_flag:
        kxop = -kxop[::-1]
        kyop = -kyop[::-1]

    # replicate across interleaves with int_rotation (can be negative)
    flag_sign = -1 if int_rotation < 0 else 1
    int_rotation = abs(int_rotation)

    phi = 2.0 * np.pi / nint
    kx = np.zeros((nk, nint), dtype=float)
    ky = np.zeros((nk, nint), dtype=float)
    for ii in range(nint):
        ang_rot = phi * flag_sign * (ii * int_rotation - (nint - 1) * np.floor(ii * int_rotation / nint))
        kx[:, ii] = kxop * np.cos(ang_rot) + kyop * np.sin(ang_rot)
        ky[:, ii] = kyop * np.cos(ang_rot) - kxop * np.sin(ang_rot)

    kx = kx.reshape(-1)
    ky = ky.reshape(-1)

    # gradients are optional — we return None to keep it light
    return kx, ky, None, None
