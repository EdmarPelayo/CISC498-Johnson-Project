import numpy as np
import os, sys
sys.path.append(os.path.abspath("."))

from src.sense.sense_operator import SenseOperator

def test_sense_forward_adjoint_shapes():
    nx, ny, nc = 32, 32, 4
    smaps = np.ones((nx, ny, nc), dtype=np.complex64)

    S = SenseOperator(smaps)

    x = np.ones(nx * ny, dtype=np.complex64)
    y = S.forward(x)

    assert y.shape == (nx * ny * nc,)
    xb = S.adjoint(y)

    assert xb.shape == (nx * ny,)
