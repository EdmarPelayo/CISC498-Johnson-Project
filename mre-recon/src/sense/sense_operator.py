import numpy as np

class SenseOperator:
    """
    Pure SENSE coil multiplication operator.

    forward:
        x (ny, nx)  →  concat over coils → (ncoils * ny * nx)
    adjoint:
        y (ncoils * ny * nx) → coil-combined adjoint image (ny, nx)
    """

    def __init__(self, coil_maps: np.ndarray):
        """
        coil_maps: (ncoils, ny, nx)
        """
        if coil_maps.ndim != 3:
            raise ValueError("coil_maps must be shape (ncoils, ny, nx)")

        self.coil_maps = coil_maps.astype(np.complex64)
        self.ncoils, self.ny, self.nx = coil_maps.shape
        self.nvox = self.ny * self.nx

        # operator shape: maps image → stacked coil images
        self.shape = (self.ncoils * self.nvox, self.nvox)

    def forward(self, x: np.ndarray):
        """
        x: (ny, nx) or flattened (nvox,)
        return: (ncoils * ny * nx,)
        """
        if x.ndim == 1:
            x = x.reshape(self.ny, self.nx)

        # multiply each coil: (ncoils, ny, nx)
        y = self.coil_maps * x[None, :, :]

        return y.reshape(-1)

    def adjoint(self, y: np.ndarray):
        """
        y: (ncoils * ny * nx,)
        returns: combined image (ny, nx)
        """
        y = y.reshape(self.ncoils, self.ny, self.nx)

        # sum coil-conjugate * data
        result = np.sum(np.conj(self.coil_maps) * y, axis=0)
        return result
