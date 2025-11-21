# src/timeseg/int_tim_seg.py
import numpy as np


def int_tim_seg(tt, L, we, seg_type="exact", we_histo=None):
    """
    Time-segmentation interpolation (min-max or histogram approximation).

    Parameters
    ----------
    tt : array_like
        Time vector (seconds), shape (Nd,).
    L : int
        Number of time segments.
    we : array_like
        Field map (rad/s), shape (Nvox,).
    seg_type : {'exact', 'histogram'}
        'exact'  -> type=1 in MATLAB (min-max using full field map).
        'histogram' -> type=2 in MATLAB (histogram-based approximation).
    we_histo : np.ndarray, optional
        (Nbins, 2) [bin_centers, counts]; required if seg_type='histogram'.

    Returns
    -------
    AA : np.ndarray (complex64)
        (L+1, Nd) interpolation weights.
    """
    tt = np.asarray(tt, dtype=np.float64).reshape(-1)
    we = np.asarray(we, dtype=np.float64).reshape(-1)
    ndat = tt.size
    N = we.size

    if L == 0:
        return np.ones((1, ndat), dtype=np.complex64)

    # Time span and segment length
    mint = tt.min()
    maxt = tt.max()
    rangt = maxt - mint
    tau = (rangt + np.finfo(float).eps) / L

    tt_shift = tt - mint
    AA = np.zeros((L + 1, ndat), dtype=np.complex64)

    if seg_type.lower() == "exact":
        # ----- type == 1 in MATLAB -----
        # Build G from exp(i * we * tau)
        phi = np.exp(1j * we[:, None] * tau)      # (N, 1)
        powers = np.arange(1, L + 1, dtype=np.float64)[None, :]  # (1, L)
        gl = phi ** powers                         # (N, L)
        G = np.concatenate([np.ones((N, 1), dtype=np.complex128), gl], axis=1)  # (N, L+1)

        glsum = gl.sum(axis=0)  # (L,)

        # Build GTG (Toeplitz-ish structure)
        GTG = np.zeros((L + 1, L + 1), dtype=np.complex128)
        GTG += np.diag(N * np.ones(L + 1, dtype=np.complex128), k=0)
        for kk in range(1, L + 1):
            v = glsum[kk - 1]
            GTG += np.diag(v * np.ones(L + 1 - kk, dtype=np.complex128), k=kk)
            GTG += np.diag(np.conjugate(v) * np.ones(L + 1 - kk, dtype=np.complex128), k=-kk)

        # Invert / pseudo-invert
        try:
            GTG_inv = np.linalg.inv(GTG)
        except np.linalg.LinAlgError:
            GTG_inv = np.linalg.pinv(GTG)

        iGTGGT = GTG_inv @ np.conjugate(G).T  # (L+1, N)

        for yy in range(ndat):
            cc = np.exp(1j * we * tt_shift[yy])  # (N,)
            AA[:, yy] = (iGTGGT @ cc).astype(np.complex64)

        return AA

    elif seg_type.lower() == "histogram":
        # ----- type == 2 in MATLAB -----
        if we_histo is None:
            raise ValueError("we_histo must be provided for seg_type='histogram'")

        we_histo = np.asarray(we_histo, dtype=np.float64)
        bin_centers = we_histo[:, 0]
        N_ap = we_histo[:, 1].astype(np.complex128)  # (Nbins,)

        rangwe = bin_centers.max() - bin_centers.min()
        minwe = bin_centers.min()
        num_bins = bin_centers.size

        # KK = floor(2*pi / ((rangwe/num_bins) * tau))
        KK = int(np.floor(2 * np.pi / ((rangwe / num_bins) * tau)))

        # Fallback to exact if KK is too small
        if KK < (2 * L + 1):
            print("int_tim_seg: KK < 2*L+1, falling back to exact interpolator.")
            return int_tim_seg(tt, L, we, seg_type="exact")

        dwn = 2 * np.pi / (KK * tau)

        k_idx = np.arange(num_bins, dtype=np.float64)  # 0..Nbins-1
        # ftwe_ap = fft(N_ap .* exp(i*dwn*[0:(num_bins-1)]*tau*(L)), KK)
        ftwe_ap = np.fft.fft(N_ap * np.exp(1j * dwn * k_idx * tau * L), KK)

        j_idx = np.arange(KK, dtype=np.float64)
        ftwe_ap *= np.exp(-1j * (minwe + dwn / 2) * tau * (j_idx - L))

        GTGap_ap = np.zeros((L + 1, L + 1), dtype=np.complex128)
        # kk MATLAB index 1..(2L+1) → python 0..(2L)
        for kk in range(2 * L + 1):
            diag_idx = kk - L  # offset from main diagonal
            v = ftwe_ap[kk]
            length = (L + 1) - abs(diag_idx)
            if length <= 0:
                continue
            GTGap_ap += np.diag(v * np.ones(length, dtype=np.complex128), k=diag_idx)

        # Invert transpose, like MATLAB's GTGap_ap.'
        try:
            iGTGap_ap = np.linalg.inv(GTGap_ap.T)
        except np.linalg.LinAlgError:
            iGTGap_ap = np.linalg.pinv(GTGap_ap.T)

        for yy in range(ndat):
            # ftc_ap = fft(N_ap.*(exp(i*[0:(num_bins-1)]*dwn*tt(yy))),KK);
            ftc_ap = np.fft.fft(N_ap * np.exp(1j * k_idx * dwn * tt_shift[yy]), KK)
            ftc_ap *= np.exp(1j * (minwe + dwn / 2) * (tt_shift[yy] - tau * j_idx))
            GTc_ap = ftc_ap[: L + 1]
            AA[:, yy] = (iGTGap_ap @ GTc_ap).astype(np.complex64)

        return AA

    else:
        raise ValueError(f"Unknown seg_type '{seg_type}'. Use 'exact' or 'histogram'.")
