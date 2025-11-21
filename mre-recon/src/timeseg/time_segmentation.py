# src/timeseg/time_segmentation.py
import numpy as np
from .int_tim_seg import int_tim_seg
from .int_tim_seghanning import int_tim_seghanning


class TimeSegmentation:
    """
    Python analog of the MATLAB TimeSegmentation class.

    Wraps a linear operator G (NUFFT-like) and applies time-segmented
    field correction in the forward / adjoint operations.

    Parameters
    ----------
    G : object
        Linear operator with:
            - shape -> (nd, np)
            - forward(x)  -> y
            - adjoint(y)  -> x
    timing_vec : array_like
        1D time vector (seconds) for **one shot**.
    field_map : array_like
        2D/3D field map (rad/s), same shape as image support.
    L : int
        Number of time segments. Use 0 for "no field correction".
    interpolator : {'minmax', 'histo', 'synthetic_histo', 'hanning'}
        Selects how temporal interpolator is computed.
    we_histo : np.ndarray, optional
        (Nbins, 2) histogram (for interpolator='histo').

    Notes
    -----
    - nShots is inferred from G.shape[0] and len(timing_vec):
        nShots = nd / len(timing_vec)
    """

    def __init__(
        self,
        G,
        timing_vec,
        field_map,
        L,
        interpolator="minmax",
        we_histo=None,
    ):
        self.G = G
        self.timing_vec = np.asarray(timing_vec, dtype=np.float64).reshape(-1)
        self.field_map = np.asarray(field_map, dtype=np.float64)
        self.L = int(L)
        self.interpolator = interpolator.lower()
        self.we_histo = we_histo

        nd, _ = self.G.shape
        shot_len = self.timing_vec.size
        if nd % shot_len != 0:
            raise ValueError(
                f"G.shape[0]={nd} not divisible by len(timing_vec)={shot_len}"
            )
        self.nShots = nd // shot_len

        # If field map is all zeros, override L
        if np.allclose(self.field_map, 0):
            print("TimeSegmentation: field map is zero → L forced to 0 (no correction).")
            self.L = 0

        # Precompute temporal interpolator (for one shot)
        if self.L == 0:
            self.time_interp = np.ones((1, shot_len), dtype=np.complex64)
        else:
            if self.interpolator == "minmax":
                we_flat = self.field_map.reshape(-1)
                self.time_interp = int_tim_seg(
                    self.timing_vec, self.L, we_flat, seg_type="exact"
                )
            elif self.interpolator == "histo":
                we_flat = self.field_map.reshape(-1)
                if self.we_histo is None:
                    raise ValueError("we_histo must be provided for interpolator='histo'")
                self.time_interp = int_tim_seg(
                    self.timing_vec, self.L, we_flat, seg_type="histogram", we_histo=self.we_histo
                )
            elif self.interpolator == "synthetic_histo":
                # Build synthetic flat histogram over reasonable range (like MATLAB)
                max_range = 2 * np.pi / (0.5e-3)  # as in original comments
                bin_centers = np.linspace(-max_range, max_range, 256)
                bin_counts = np.ones_like(bin_centers)
                we_histo = np.stack([bin_centers, bin_counts], axis=1)
                we_flat = self.field_map.reshape(-1)
                self.time_interp = int_tim_seg(
                    self.timing_vec, self.L, we_flat, seg_type="histogram", we_histo=we_histo
                )
            elif self.interpolator == "hanning":
                self.time_interp = int_tim_seghanning(self.timing_vec, self.L)
            else:
                raise ValueError(f"Unknown interpolator '{self.interpolator}'")

        self._is_transpose = False

    # Like MATLAB ctranspose: toggle transpose flag
    def T(self):
        obj = TimeSegmentation(
            self.G,
            self.timing_vec.copy(),
            self.field_map.copy(),
            self.L,
            interpolator=self.interpolator,
            we_histo=self.we_histo,
        )
        obj.time_interp = self.time_interp
        obj.nShots = self.nShots
        obj._is_transpose = not self._is_transpose
        return obj

    @property
    def shape(self):
        return self.G.shape

    # Forward / adjoint operations
    def __matmul__(self, x):
        """
        Use `A @ x` as forward (or adjoint) operation.
        """
        if self._is_transpose:
            return self._adjoint(x)
        else:
            return self._forward(x)

    # Or explicit methods if you prefer
    def forward(self, x):
        return self._forward(x)

    def adjoint(self, y):
        return self._adjoint(y)

    # --- internal helpers ---

    def _forward(self, x):
        """
        y = A * x  (field corrected forward model)
        """
        x = np.asarray(x, dtype=np.complex64).reshape(-1)

        if self.L == 0:
            # No field correction, just apply phase at minTime and NUFFT
            minTime = self.timing_vec.min()
            phase0 = np.exp(-1j * self.field_map.reshape(-1) * minTime)
            return self.G.forward(phase0 * x)

        tau = (self.timing_vec.max() - self.timing_vec.min() + np.finfo(float).eps) / self.L
        minTime = self.timing_vec.min()
        nd, _ = self.G.shape

        # Global phase
        x_mod = np.exp(-1j * self.field_map.reshape(-1) * minTime) * x

        # (nd, L+1)
        y_temp = np.zeros((nd, self.L + 1), dtype=np.complex64)

        for ii in range(self.L + 1):
            Wo = np.exp(-1j * self.field_map.reshape(-1) * (ii * tau))
            y_temp[:, ii] = self.G.forward(Wo * x_mod)

        # replicate temporal interpolator across shots
        # time_interp: (L+1, shot_len)
        shot_len = self.timing_vec.size
        aa = np.tile(self.time_interp.T, (self.nShots, 1))  # (nd, L+1)

        y = np.sum(y_temp * aa.astype(np.complex64), axis=1)
        return y

    def _adjoint(self, y):
        """
        x = A' * y  (field corrected adjoint model)
        """
        y = np.asarray(y, dtype=np.complex64).reshape(-1)
        nd, npix = self.G.shape

        if self.L == 0:
            # No field correction: just adjoint + global phase
            minTime = self.timing_vec.min()
            x = self.G.adjoint(y)
            phase = np.exp(1j * self.field_map.reshape(-1) * minTime)
            return phase * x

        tau = (self.timing_vec.max() - self.timing_vec.min() + np.finfo(float).eps) / self.L

        x_temp = np.zeros((npix, self.L + 1), dtype=np.complex64)
        shot_len = self.timing_vec.size

        for ii in range(self.L + 1):
            Wo = np.exp(1j * self.field_map.reshape(-1) * (ii * tau))
            aa = np.tile(self.time_interp[ii, :], self.nShots)  # (nd,)
            x_temp[:, ii] = Wo * self.G.adjoint(aa.astype(np.complex64) * y)

        x = np.sum(x_temp, axis=1)
        minTime = self.timing_vec.min()
        phase = np.exp(1j * self.field_map.reshape(-1) * minTime)
        return phase * x
