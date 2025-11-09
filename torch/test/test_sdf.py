# dev tests for pytorch signed distance function
import unittest
import torch
import torch.nn.functional as F
import numpy as np
from torch.func import jacfwd, jacrev, vmap
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt as edt
from scipy.ndimage import gaussian_filter

from scalar_field_interpolator import ScalarFieldInterpolator, SDF

class MyTestCase(unittest.TestCase):

    def test_edt(self):
        # we need smooth (negative) SDT even inside obstacles
        occupancy = np.zeros((51,51), dtype=bool)
        occupancy[25:51,:] = 1
        plt.figure()
        plt.imshow(occupancy)
        plt.show()
        plt.figure()
        # these two factors account for outside and
        # inside obstacles
        sdf = edt(~occupancy) + -1.0*edt(occupancy)
        plt.imshow(sdf)
        plt.colorbar()
        plt.show()

    def test_room(self):
        sdf = SDF(1, 1, 0.5, 0.5, .02)
        sdf.generate(0.14, 0.14)

    def test_sample(self):
        sdf = SDF(3, 2, -1.5, -1.0, .02)
        sdf.generate(0.14, 0.14)

        sfi = ScalarFieldInterpolator(sdf.sdf, sdf.ox, sdf.oy, sdf.res)

        dx = 0.05
        x = np.arange(sdf.ox, sdf.ox+sdf.x_size + dx, dx)
        y = np.arange(sdf.oy, sdf.oy+sdf.y_size + dx, dx)
        xx, yy = np.meshgrid(x, y)

        s_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        u_pts = np.zeros(s_pts.shape)
        S = torch.from_numpy(s_pts)
        U = torch.from_numpy(u_pts)

        c = vmap(sfi.interpolator, in_dims=(0, 0))(S, U)  # (T,)
        c_np = c.detach().cpu().numpy()
        c_np = np.reshape(c_np, (41,61))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(c_np,
                   origin='lower',
                   extent=[x.min(), x.max(), y.min(), y.max()],  # map array to coordinate bounds
                   aspect='auto',  # or 'equal'
                   cmap='viridis'
                   )
        plt.colorbar()
        plt.show()

    def test_jacobian(self):
        sdf = SDF(2, 2, -1, -1, .02)
        sdf.generate(0.14, 0.14)

        sfi = ScalarFieldInterpolator(sdf.sdf, sdf.ox, sdf.oy, sdf.res)

        s_pts = np.array([[-0.5, -0.5, 42], [0.0, -0.5, 43], [-0.5, 0.0, 44], [0.0, 0.0, 45]],dtype=np.float32)
        u_pts = np.array([[0.0], [0.0], [0.0], [0.0]], dtype=np.float32)

        S = torch.from_numpy(s_pts)
        U = torch.from_numpy(u_pts)

        jac_fun = jacrev(sfi.interpolator, argnums=(0, 1))
        c = vmap(sfi.interpolator, in_dims=(0, 0))(S, U)  # (T,)
        A, B = vmap(jac_fun, in_dims=(0, 0))(S, U)  # A: (T,4), B: (T,1)

        d = c - torch.einsum('tij,tj->ti', A, S) - torch.einsum('tij,tj->ti', B, U)
        # back to numpy
        d_ret = d.detach().cpu().numpy()
        A_ret = A.detach().cpu().numpy()
        B_ret = B.detach().cpu().numpy()

        print(d_ret)
        print(A_ret)
        print(B_ret)


if __name__ == '__main__':
    unittest.main()
