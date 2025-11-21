import numpy as np

class SenseOperator:
    """
    SENSE encoding operator:
        y = sum_coils( coil * image )
        x = sum_coils( coil^H * image )
    """
    def __init__(self, smaps):
        """
        smaps: (nx, ny, nc) coil sensitivity maps
        """
        assert smaps.ndim == 3, "smaps must be (nx, ny, nc)"

        self.smaps = smaps.astype(np.complex64)
        self.nx, self.ny, self.nc = smaps.shape
        self.n_vox = self.nx * self.ny

    @property
    def shape(self):
        """
        Returns (n_data, n_vox).
        """
        return (self.n_vox * self.nc, self.n_vox)

    def forward(self, x):
        """
        x: (n_vox,) image
        returns (n_vox * nc,) with coil sensitivities applied
        """
        img = x.reshape(self.nx, self.ny)
        out = np.zeros((self.nx, self.ny, self.nc), dtype=np.complex64)

        for c in range(self.nc):
            out[..., c] = img * self.smaps[..., c]

        return out.reshape(-1)

    def adjoint(self, y):
        """
        y: (n_vox * nc,) coil images
        returns: (n_vox,) combined image
        """
        y = y.reshape(self.nx, self.ny, self.nc)
        img = np.zeros((self.nx, self.ny), dtype=np.complex64)

        for c in range(self.nc):
            img += y[..., c] * np.conj(self.smaps[..., c])

        return img.reshape(-1)
