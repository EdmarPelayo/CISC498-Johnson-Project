import numpy as np


def genspivd_kim(D, N, nl=1, gamp=4.0, gslew=400.0, gts=10e-6, alphavd=4):
    """
    Python port of genspivd_Kim.m

    Variable-density spiral gradient design:
      - Duyn’s slew-limited design
      - Glover-style analytic spiral
      - Two-phase time warping (tau1, tau2)
      - VD exponent alphavd

    Parameters
    ----------
    D : float
        FOV in cm
    N : int
        matrix size
    nl : int
        number of interleaves
    gamp : float
        gradient amplitude limit [G/cm]
    gslew : float
        slew-rate limit [mT/m/ms]
    gts : float
        gradient raster / output sample spacing [s]
    alphavd : float
        VD exponent

    Returns
    -------
    Gx, Gy : gradients
    kx, ky : k-space trajectory
    sx, sy : slew rates
    """

    # -----------------------------------------------
    # Constants
    # -----------------------------------------------
    GRESMAX = 21000
    gamma = 4.257e3  # Hz/G

    S0 = gslew * 100           # approximate conversion to G/cm/ms
    rdt = gts * 0.5            # internal finer time step

    # -----------------------------------------------
    # k-space target radius
    # -----------------------------------------------
    km = 0.5 * N / D  # 1/cm

    # -----------------------------------------------
    # Number of turns (n)
    # -----------------------------------------------
    nturn = int(np.floor(1 / (1 - (1 - 2/(N/nl)) ** (1/alphavd))))
    omega = 2 * np.pi * nturn

    # -----------------------------------------------
    # Compute key times
    # -----------------------------------------------
    Te = np.sqrt(S0 * gamma / (km * omega**2))
    Tr = (gamma * gamp) / (km * omega)

    Ts = 1 / ((alphavd/2 + 1) * Te)
    T1 = (alphavd/2 + 1) * Te

    ts = (Tr * (alphavd/2 + 1) /
          (T1 ** ((alphavd + 1) / (alphavd*0.5 + 1)))) ** (1 + 2/alphavd)

    # -----------------------------------------------
    # Build tau(t)
    # -----------------------------------------------
    if ts < Ts:
        # First segment: slew-limited
        t1 = np.arange(0, ts + rdt/2, rdt)
        tau1 = ((alphavd/2 + 1) *
                np.sqrt((S0*gamma) / (km * omega**2)) * t1) ** (1/(alphavd/2 + 1))

        # Quantize ts to match MATLAB behavior
        ts_q = np.floor(ts / gts) * gts

        # Transition times
        tm = (Te * (alphavd/2 + 1) * ts_q) ** ((alphavd + 1)/(alphavd*0.5 + 1)) \
             / (Tr * (alphavd + 1))

        Ts = 1 / (Tr * (alphavd + 1))

        # Second segment
        t2 = np.arange(tm + rdt, Ts + rdt/2, rdt)
        tau2 = ((gamma * gamp) / (km * omega) *
                (alphavd+1) * t2) ** (1/(alphavd + 1))

        tau = np.concatenate([tau1, tau2])

    else:
        # Only one segment
        t = np.arange(0, Ts + rdt/2, rdt)
        tau = ((alphavd/2 + 1) *
               np.sqrt((S0*gamma) / (km * omega**2)) * t) ** (1/(alphavd/2 + 1))

    # -----------------------------------------------
    # Build k-space path
    # -----------------------------------------------
    k = km * (tau ** alphavd) * np.exp(1j * omega * tau)

    # -----------------------------------------------
    # Gradient
    # -----------------------------------------------
    gx_complex = np.diff(k) / (gamma * rdt)

    # Downsample to gradient raster points
    Gx = np.real(gx_complex[::2])
    Gy = np.imag(gx_complex[::2])

    # -----------------------------------------------
    # Slew rate
    # -----------------------------------------------
    g_complex = Gx + 1j * Gy
    s_complex = np.diff(g_complex) / (gts * 1000)  # per ms

    sx = np.real(s_complex)
    sy = np.imag(s_complex)

    # -----------------------------------------------
    # Integrate to get k-space
    # -----------------------------------------------
    Kx = np.cumsum(np.concatenate([[0], Gx])) * gts * D * gamma
    Ky = np.cumsum(np.concatenate([[0], Gy])) * gts * D * gamma

    return Gx, Gy, Kx, Ky, sx, sy
