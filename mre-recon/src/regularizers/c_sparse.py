import numpy as np
from scipy import sparse

def c_sparse(mask, dims2penalize=None):
    """
    Python translation of MATLAB C_sparse.m

    Creates a roughness penalty matrix C and weight vector wt for the mask image.
    Each row of C represents a finite difference (1 - shifted voxel)
    along one dimension. Used for smoothness regularization in image reconstruction.

    Parameters
    ----------
    mask : np.ndarray
        N-D binary (0/1) or integer array specifying active voxels.
    dims2penalize : list or np.ndarray, optional
        1-based list of dimensions to penalize (default: all).

    Returns
    -------
    C : scipy.sparse.csr_matrix
        Sparse penalty matrix (M x N_active), where each row is a difference.
    wt : np.ndarray
        Weight vector for the rows of C (0 where difference crosses mask boundary).
    """
    sz_img = np.array(mask.shape)
    dims = len(sz_img)
    N = mask.size

    if dims2penalize is None:
        dims2penalize = np.arange(1, dims + 1)

    # Identity sparse (equivalent to spdiag(ones))
    C_onesSp = sparse.identity(N, format='csr')

    C_blocks = []
    wt_list = []
    shift_val = 1  # MATLAB shift unit (column-major order)

    for ii in range(dims):
        # Circular shift along flattened (column-major) array
        i_indices = np.arange(N)
        j_indices = np.roll(i_indices, shift_val)
        data = np.ones(N)
        C_shiftSp = sparse.csr_matrix((data, (i_indices, j_indices)), shape=(N, N))
        C_dim = C_onesSp - C_shiftSp
        C_blocks.append(C_dim)

        # Boundary weights
        wt_discard = np.zeros(shift_val)
        wt_keep = np.ones(shift_val * (sz_img[ii] - 1))
        wt_tmp = np.concatenate([wt_discard, wt_keep])
        shift_val *= sz_img[ii]

        num_repeats = N // shift_val
        wt_array = np.tile(wt_tmp, num_repeats)

        # Apply voxel mask (column-major flattening to match MATLAB)
        wt_array = wt_array * mask.flatten(order='F')
        wt_list.append(wt_array)

    # Stack differences
    C_raw = sparse.vstack(C_blocks, format='csr')
    wt_full = np.concatenate(wt_list)

    # Filter dimensions according to dims2penalize (1-based)
    rows_to_keep = np.zeros(C_raw.shape[0], dtype=bool)
    n_rows_per_dim = N
    for i_dim in range(dims):
        start = i_dim * n_rows_per_dim
        end = (i_dim + 1) * n_rows_per_dim
        if (i_dim + 1) in dims2penalize:
            rows_to_keep[start:end] = True
    C_filtered = C_raw[rows_to_keep, :]
    wt_filtered = wt_full[rows_to_keep]

    # Apply weights to zero out boundaries/masked voxels
    C_weighted = sparse.diags(wt_filtered) @ C_filtered

    # Keep only unmasked voxels as columns
    mask_flat = mask.flatten(order='F').astype(bool)
    C_masked = C_weighted[:, mask_flat]

    # Zero rows that cross mask boundary (MATLAB sets to zero, not remove)
    row_sum = np.array(C_masked.sum(axis=1)).flatten()
    idx_to_zero = (row_sum != 0)
    if np.any(idx_to_zero):
        # Make modifiable CSR copy
        C_masked = C_masked.tolil()
        C_masked[idx_to_zero, :] = 0
        C_masked = C_masked.tocsr()

    wt = wt_filtered
    return C_masked, wt
