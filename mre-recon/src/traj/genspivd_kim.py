# genspivd_kim.py
# NumPy port of genspivd_Kim.m (variable-density spiral with alpha_vd)
import numpy as np

def genspivd_kim(D, N, nl=1, gamp=4.0, gslew=400.0, gts=10e-6, alphavd=4.0):
    """
    Variable-density spiral design (Kim), matching MATLAB flow.

    Returns
    -------
    Gx, Gy : gradients [G/cm] (decimated 1:2 as in MATLAB)
    kx, ky : k-space trajectory [1/cm]
    sx, sy : slew (finite diff) [G/cm/s] scaled like MATLAB
    """
    GRESMAX = 21000
    gamma = 4.257e3       # Hz/G (MATLAB used gambar-like here)
    rdt = gts * 0.5
    S0 = gslew * 100.0
    G0 = gamp

    km = 0.5 * N / D      # max k radius [1/cm]
    n = int(np.floor(1.0 / (1.0 - (1.0 - 2.0/(N/nl))**(1.0/alphavd))))
    n = max(n, 1)
    omega = 2.0 * np.pi * n

    Te = np.sqrt(S0 * gamma / (km * omega**2))
    Tr = (gamma * G0) / (km * omega)

    Ts = 1.0 / ((alphavd/2.0 + 1.0) * Te)
    ts = (Tr*(alphavd/2.0+1.0) / ( (Te*(alphavd/2.0+1.0))**((alphavd+1.0)/(alphavd*0.5+1.0)) ))**(1.0 + 2.0/alphavd)

    if ts < Ts:
        t1 = np.arange(0, ts + 1e-12, rdt)
        tau1 = ((alphavd/2.0 + 1.0) * np.sqrt((S0*gamma)/(km*omega**2)) * t1)**(1.0/(alphavd/2.0+1.0))
        # quantize like MATLAB
        ts_q = np.floor(ts / gts) * gts
        # recompute second segment with constant-G
        # find tm, Ts from MATLAB code (kept consistent)
        # These “tm, Ts” expressions in original MATLAB are a bit odd; we keep same form:
        tm = (Te*(alphavd/2.0+1.0)*ts_q)**((alphavd+1.0)/(alphavd*0.5+1.0)) / (Tr*(alphavd+1.0))
        Ts2 = 1.0 / (Tr*(alphavd+1.0))
        t2 = np.arange(tm + rdt, Ts2 + 1e-12, rdt)
        tau2 = ((gamma * G0)/(km*omega) * (alphavd+1.0) * t2)**(1.0/(alphavd+1.0))
        tau = np.concatenate([tau1, tau2])
        T = Ts2
    else:
        t = np.arange(0, Ts + 1e-12, rdt)
        tau = ((alphavd/2.0 + 1.0) * np.sqrt((S0*gamma)/(km*omega**2)) * t)**(1.0/(alphavd/2.0+1.0))
        T = Ts

    k = km * (tau**alphavd) * np.exp(1j * omega * tau)
    gx = np.diff(k) / (gamma * rdt)  # gradient [G/cm] (complex)
    Gx_full = np.real(gx)
    Gy_full = np.imag(gx)

    # decimate like MATLAB (1:2:l2)
    Gx = Gx_full[::2]
    Gy = Gy_full[::2]

    # integrate to k-space (match MATLAB scaling)
    Kx = np.cumsum(np.concatenate([[0.0], Gx])) * gts * D * gamma
    Ky = np.cumsum(np.concatenate([[0.0], Gy])) * gts * D * gamma
    kx = np.real(Kx)
    ky = np.real(Ky)


    g = Gx + 1j * Gy
    s = np.diff(g) / (gts * 1000.0)
    sx = np.real(s)
    sy = np.imag(s)

    return Gx, Gy, kx, ky, sx, sy
