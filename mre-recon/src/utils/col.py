import numpy as np
def col(x):
    """
    "colon" function: flattens x into a single column vector.
    
    This emulates MATLAB's x(:) operator.
    - .reshape(-1, 1) forces the array into a 2D column vector.
    - order='F' (Fortran order) specifies column-major flattening,
      which is MATLAB's default.
    """
    # Ensure x is a numpy array for the .reshape method
    x_np = np.asarray(x) 
    return x_np.reshape(-1, 1, order='F')