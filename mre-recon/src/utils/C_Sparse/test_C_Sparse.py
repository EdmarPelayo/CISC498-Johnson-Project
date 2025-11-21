import unittest
import numpy as np
from scipy import sparse
from C_sparse import c_sparse

class TestCSparse(unittest.TestCase):

    def test_2d_mask_basic(self):
        mask = np.ones((3, 3), dtype=bool)
        C, wt = c_sparse(mask)

        self.assertIsInstance(C, sparse.spmatrix)
        self.assertEqual(C.shape[1], mask.size)
        self.assertEqual(wt.shape[0], C.shape[0])
        self.assertFalse(np.any(np.isnan(wt)))
        self.assertTrue(np.all(wt >= 0))

    def test_3d_mask(self):
        mask = np.ones((3, 3, 3), dtype=bool)
        C, wt = c_sparse(mask)
        self.assertEqual(C.shape[1], mask.size)
        self.assertEqual(wt.shape[0], C.shape[0])
        self.assertLess(C.nnz, np.prod(C.shape))  # sparse check

    def test_dims_to_penalize_partial(self):
        mask = np.ones((4, 4), dtype=bool)
        dims2penalize = [1, 0]  # disable y-axis penalization
        C, wt = c_sparse(mask, dims2penalize)
        self.assertTrue(np.any(wt == 0))

    def test_all_zero_mask(self):
        mask = np.zeros((3, 3), dtype=bool)
        C, wt = c_sparse(mask)
        self.assertEqual(C.nnz, 0)
        self.assertTrue(np.all(wt == 0))

    def test_1d_mask(self):
        mask = np.ones((4,), dtype=bool)
        C, wt = c_sparse(mask)
        self.assertGreaterEqual(C.shape[0], len(mask) - 1)
        self.assertIsInstance(C, sparse.spmatrix)

    def test_deterministic(self):
        np.random.seed(0)
        mask = np.random.randint(0, 2, (4, 4), dtype=bool)
        C1, wt1 = c_sparse(mask)
        C2, wt2 = c_sparse(mask)
        self.assertTrue((C1 != C2).nnz == 0)
        np.testing.assert_array_equal(wt1, wt2)


if __name__ == '__main__':
    unittest.main()
