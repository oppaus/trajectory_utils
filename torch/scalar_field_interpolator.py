# pytorch signed distance function
# emitted by GPT5 and simplified/cleaned up

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from scipy.ndimage import gaussian_filter

class SDF:
    x_size: float
    y_size: float
    ox: float
    oy: float
    res: float
    h: int
    w: int
    sdf: np.ndarray
    def __init__(self, x_size: float, y_size: float, origin_x: float, origin_y: float, resolution: float):
        self.x_size = x_size
        self.y_size = y_size
        self.ox = origin_x
        self.oy = origin_y
        self.res = resolution
        self.h = np.round(y_size / resolution).astype(int)
        self.w = np.round(x_size / resolution).astype(int)
        self.sdf = np.array([])

    def generate(self, robot_radius: float, obstacle_radius: float):
        """Generate signed distance field for the obstacle constraints."""
        occupancy = np.zeros((self.h, self.w), dtype=bool)
        occupancy[:, 0] = 1
        occupancy[:, -1] = 1
        occupancy[0, :] = 1
        occupancy[-1, :] = 1
        occupancy[int(np.round(self.h/2.0)), int(np.round(3.0 * self.w / 4.0))] = 1
        # we need smooth SDF inside obstacles!
        dist = (edt(~occupancy) - 1.0*edt(occupancy)) * self.res
        sdf = dist - (robot_radius + obstacle_radius)
        # sigma units are pixels - um res units
        self.sdf = gaussian_filter(sdf, sigma=4.0).astype(np.float32)



class ScalarFieldInterpolator:
    def __init__(self, phi_np: np.ndarray, origin_x: float, origin_y: float, resolution: float):
        # (H,W) tensor (float32/64)
        self.phi = torch.from_numpy(phi_np)
        self.ox = origin_x
        self.oy = origin_y
        self.res = resolution
    
    def bilinear_sample(self,
        xy: torch.Tensor,
        padding_mode: str = "border",
    ) -> torch.Tensor:
        """
        Returns values at xy: (T,) in coordinates defined by origin and resolution
        Works with autograd (reverse and forward AD).
        """
        H, W = self.phi.shape
        x = xy[:, 0]
        y = xy[:, 1]
        x_pix = (x - self.ox) / self.res
        y_pix = (y - self.oy) / self.res
    
        # corners
        x0 = torch.floor(x_pix).to(torch.long)
        y0 = torch.floor(y_pix).to(torch.long)
        x1 = x0 + 1
        y1 = y0 + 1
    
        if padding_mode == "border":
            x0 = x0.clamp(0, W - 1)
            x1 = x1.clamp(0, W - 1)
            y0 = y0.clamp(0, H - 1)
            y1 = y1.clamp(0, H - 1)
    
            Ia = self.phi[y0, x0]
            Ib = self.phi[y0, x1]
            Ic = self.phi[y1, x0]
            Id = self.phi[y1, x1]
    
            wx = (x_pix - x0.to(x_pix.dtype))
            wy = (y_pix - y0.to(y_pix.dtype))
    
            vals = (Ia * (1 - wx) * (1 - wy) +
                    Ib *       wx  * (1 - wy) +
                    Ic * (1 - wx) *       wy  +
                    Id *       wx  *       wy)
    
            return vals
    
        elif padding_mode == "zeros":
            # mask in-bounds
            in_x0 = (x0 >= 0) & (x0 < W)
            in_x1 = (x1 >= 0) & (x1 < W)
            in_y0 = (y0 >= 0) & (y0 < H)
            in_y1 = (y1 >= 0) & (y1 < H)
    
            def safe_get(xx, yy):
                mask = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
                out = torch.zeros_like(x_pix)
                valid = mask.nonzero(as_tuple=False).squeeze(-1)
                if valid.numel() > 0:
                    out[valid] = self.phi[yy[valid], xx[valid]]
                return out
    
            Ia = safe_get(x0, y0)
            Ib = safe_get(x1, y0)
            Ic = safe_get(x0, y1)
            Id = safe_get(x1, y1)
    
            wx = (x_pix - x0.to(x_pix.dtype))
            wy = (y_pix - y0.to(y_pix.dtype))
    
            vals = (Ia * (1 - wx) * (1 - wy) +
                    Ib *       wx  * (1 - wy) +
                    Ic * (1 - wx) *       wy  +
                    Id *       wx  *       wy)
    
            return vals
        else:
            raise ValueError("padding_mode must be 'border' or 'zeros'")

    # not used, but left around to help in case gpt5 has built in some
    # sort of bug I am not expecting
    def world_to_norm_xy(self, x, y, H, W, ox, oy, res, flip_y=True, align_corners=True):
        # map world meters -> pixel -> normalized [-1,1]
        x_pix = (x - ox) / res
        y_pix = (y - oy) / res
        if flip_y:
            y_pix = (H - 1) - y_pix
        if align_corners:
            x_norm = 2.0 * (x_pix / (W - 1)) - 1.0
            y_norm = 2.0 * (y_pix / (H - 1)) - 1.0
        else:
            x_norm = (x_pix + 0.5) * (2.0 / W) - 1.0
            y_norm = (y_pix + 0.5) * (2.0 / H) - 1.0
        return torch.stack([x_norm, y_norm], dim=-1)  # (...,2)

    def interpolator(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        xy = s[:2]
        val = self.bilinear_sample(xy[None, :], padding_mode="border")
        return val  # scalar ()
