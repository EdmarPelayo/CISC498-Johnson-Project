import numpy as np

def col(x):
    """
    Flatten array into a column-like 1D vector.
    MATLAB x(:) becomes x.reshape(-1)
    """
    return np.asarray(x).reshape(-1)
