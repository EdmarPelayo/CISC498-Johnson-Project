# genspi.py
# NumPy port of genspi.m (slew-limited + gmax-limited analytic spiral)
import numpy as np

def genspi(D, N, nl=1, gamp=4.0, gslew=400.0, gts=10e-6):
    """
    Parameters
    ----------
    D : float
        FOV in cm
    N : int
        Matrix size (target resolution)
    nl : int
        Number of interleaves
    gamp : float
        Max gradient [G/cm] (≈ 4 for Allegra in original code)
    gslew : float
        Slew rate [mT/m/ms] (≈ 400 in original code)
    gts : float
        Gradient raster time [s] (e.g., 10e-6)

    Returns
    -------
    Gx, Gy : (L,) gradients [G/cm] at raster gts (every other sample, like MATLAB’s 1:2:l2)
    kx, ky : (Lk,) k-space trajectory [1/cm]
    sx, sy : (L-1,) gradient slew (finite diff) [G/cm/s] scaled like MATLAB
    """
    # constants
    GRESMAX = 210000  # large upper bound for gradient samples (kept as in MATLAB)
    gamma = 2 * np.pi * 4.257e3  # rad/s/G
    gambar = gamma / (2 * np.pi) # Hz/G

    # scalar aliases
    S0 = gslew * 100.0       # mT/m/ms -> G/cm/s (matches MATLAB scaling)
    dt = gts * 0.5
    q = 5.0

    # slew-limited approximation
    Ts = (2/3) * (1.0 / nl) * np.sqrt(((np.pi * N)**3) / (gamma * D * S0))
    # time vector (half-raster)
    t = np.arange(0, Ts + 1e-12, dt)
    x = t**(4/3)
    beta = S0 * gamma * D / nl
    a2 = (N * np.pi) / (nl * (Ts**(2/3)))
    dthdt = t * (beta * (q + (1/6) * beta/a2 * x) / (q + 0.5 * beta/a2 * x)**2)
    theta = (t**2) * (0.5 * beta / (q + 0.5 * beta/a2 * x))

    c, s = np.cos(theta), np.sin(theta)
    gx = (nl / (D * gamma)) * dthdt * (c - theta * s)
    gy = (nl / (D * gamma)) * dthdt * (s + theta * c)
    gabs = np.abs(gx + 1j*gy)

    # clip if over peak gradient
    gmax = np.abs(gamp / (theta + 1e-15) + 1j * gamp)
    l1 = len(t) - np.count_nonzero(gabs > gmax)
    l1 = max(1, l1)
    ts = t[l1-1]
    thetas = theta[l1-1]

    # gmax-limited continuation
    Gx = gx.copy()
    Gy = gy.copy()
    if np.max(gabs) > gamp:
        # extend to T such that gradient saturates at gamp
        T = ((np.pi*N/nl)**2 - thetas**2) / (2 * gamma * gamp * D / nl) + ts
        t2 = np.arange(ts + dt, T + 1e-12, dt)
        theta2 = np.sqrt(thetas**2 + (2 * gamma * gamp * D / nl) * (t2 - ts))
        c2, s2 = np.cos(theta2), np.sin(theta2)
        gx2 = gamp * (c2/theta2 - s2)
        gy2 = gamp * (s2/theta2 + c2)
        # append
        Gx = np.concatenate([gx[:l1], gx2])
        Gy = np.concatenate([gy[:l1], gy2])

    # decimate like MATLAB's 1:2:l2
    l2 = len(Gx)
    Gx = Gx[::2]
    Gy = Gy[::2]

    # k-space integrate gradients -> [1/cm]
    # (Grad [G/cm]) * gts [s] * FOV [cm] * gambar [Hz/G] -> cycles/cm; here we match MATLAB:
    Kx = np.cumsum(np.concatenate([[0.0], Gx])) * gts * D * gambar
    Ky = np.cumsum(np.concatenate([[0.0], Gy])) * gts * D * gambar
    kx = np.real(Kx)
    ky = np.real(Ky)

    # finite-difference for slew (match MATLAB scaling)
    g = Gx + 1j*Gy
    s = np.diff(g) / (gts * 1000.0)  # (/ms) like MATLAB
    sx = np.real(s)
    sy = np.imag(s)

    return Gx, Gy, kx, ky, sx, sy
