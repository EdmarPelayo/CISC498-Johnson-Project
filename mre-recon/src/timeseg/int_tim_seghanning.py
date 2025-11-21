# src/timeseg/int_tim_seghanning.py
import numpy as np


def int_tim_seghanning(tt, L):
    """
    Hanning-based time segmentation weights.

    Parameters
    ----------
    tt : array_like
        Time vector (seconds), shape (Nd,) or (Nd,1).
    L : int
        Number of time segments.

    Returns
    -------
    AA : np.ndarray (complex64)
        (L+1, Nd) array of interpolation weights.
    """
    tt = np.asarray(tt, dtype=np.float64).reshape(-1)
    ndat = tt.size

    # If no segmentation requested, just return ones
    if L == 0:
        return np.ones((1, ndat), dtype=np.complex64)

    mint = tt.min()
    maxt = tt.max()
    rangt = maxt - mint
    tau = (rangt + np.finfo(float).eps) / L

    # Shift so min time is 0 (matches MATLAB)
    tt_shift = tt - mint

    AA = np.zeros((L + 1, ndat), dtype=np.complex64)

    for ll in range(L + 1):
        center = ll * tau
        arg = tt_shift - center
        # Hanning window over |arg| < tau
        w = 0.5 + 0.5 * np.cos(np.pi * arg / tau)
        mask = (np.abs(arg) < tau)
        AA[ll, :] = (w * mask).astype(np.complex64)

    return AA
