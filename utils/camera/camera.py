import numpy as np 
import torch 
from typing import Optional

from .camera_extrinsics import CameraExtrinsics
from .camera_intrinsics import CameraIntrinsics

class Camera:
    def __init__(self, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics, name: Optional[str]=None, image_path: Optional[str]=None):
        """_summary_

        Args:
            intrinsics (CameraIntrinsics): _description_
            extrinsics (CameraExtrinsics): _description_
        """
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.name = name
        self.image_path = image_path
        
        self.R = extrinsics.R.numpy()
        self.t = extrinsics.t.numpy()
        self.K = intrinsics.get_K()
        self.Rt = np.hstack([self.R, self.t])
        self.P = self.K @ self.Rt  # Projection matrix

    
    
    def project(self, point3d: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            point3d (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        points_h = np.hstack([point3d, np.ones((point3d.shape[0], 1))])  # Convert to homogeneous coordinates
        proj = (self.P @ points_h.T).T
        uv = proj[:, :2] / proj[:, 2:3]  # Normalize by the third coordinate
        return uv
    
    def unproject(self, pixels: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            pixels (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        fx, fy = self.intrinsics.fl_x, self.intrinsics.fl_y
        cx, cy = self.intrinsics.cx, self.intrinsics.cy
        
        x = (pixels[:, 0] - cx) / fx
        y = (pixels[:, 1] - cy) / fy
        rays = np.stack([x, y, np.ones_like(x)], axis=-1)  # Direction in camera space
        rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)  # Normalize
        return rays
    
    #need to convert to torch tensors for these functions
    def to_torch(self, device="cpu", dtype=torch.float32):
        """_summary_

        Args:
            device (str, optional): _description_. Defaults to "cpu".
            dtype (_type_, optional): _description_. Defaults to torch.float32.
        """
        return{
            "K": torch.tensor(self.K, device=device, dtype=dtype),
            "R": torch.tensor(self.R, device=device, dtype=dtype),
            "t": torch.tensor(self.t, device=device, dtype=dtype),
            "Rt": torch.tensor(self.Rt, device=device, dtype=dtype),
            "P": torch.tensor(self.P, device=device, dtype=dtype)
        }
    
    def rays(self, H: int , W: int, device="cpu", dtype=torch.float32)-> tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            H (int): _description_
            W (int): _description_
            device (str, optional): _description_. Defaults to "cpu".
            dtype (_type_, optional): _description_. Defaults to torch.float32.

        Returns:
            torch.Tensor: _description_
        """
        i, j = torch.meshgrid(
            torch.arange(W, device=device, dtype=dtype),
            torch.arange(H, device=device, dtype=dtype),
            indexing='xy'
        
        )
        pixels = torch.stack([i, j], dim=-1).reshape(-1, 2)  # (H*W, 2)
        d_cam = self.unproject(pixels.cpu().numpy())  # (H*W, 3) unprojection to camera space
        d_cam = torch.tensor(d_cam, device=device, dtype=dtype)
        pos = self.extrinsics.get_position()
        if isinstance(pos, torch.Tensor):
            pos = pos.reshape(1, 3)
        else:
            pos = torch.tensor(pos, device=device, dtype=dtype).reshape(1, 3)
        d_world = self.extrinsics.camera2world(d_cam) - pos  # (H*W, 3) direction in world space
        d_world = d_world / torch.norm(d_world, dim=-1, keepdim=True)
        
        origin = pos.expand_as(d_world)#ray origins
        
        return origin, d_world
    
    #This part is inspired by the NeRF rendering approch from the orignal paper 
    def get_rays_perspective(self, H: int, W: int, device="cpu", dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        K = self.intrinsics.get_K().to(device=device, dtype=dtype)
        c2w = self.extrinsics.get_c2w().to(device=device, dtype=dtype)
        
        # Grille de pixels
        i, j = torch.meshgrid(
            torch.linspace(0, W-1, W, device=device, dtype=dtype),
            torch.linspace(0, H-1, H, device=device, dtype=dtype),
            indexing='xy'
        )
        
        dirs = torch.stack([
            (i - K[0, 2]) / K[0, 0],
            -(j - K[1, 2]) / K[1, 1],
            -torch.ones_like(i)
        ], dim=-1)  # (H, W, 3)
        
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)  # (H, W, 3)
        rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)
        
        return rays_o, rays_d
    
    def get_rays_orthographic(self, H: int, W: int, device="cpu", dtype=torch.float32, z_dir=-1.0) -> tuple[torch.Tensor, torch.Tensor]:
        
        K = self.intrinsics.get_K().to(device=device, dtype=dtype)
        c2w = self.extrinsics.get_c2w().to(device=device, dtype=dtype)
        
        i, j = torch.meshgrid(
            torch.linspace(0, W-1, W, device=device, dtype=dtype),
            torch.linspace(0, H-1, H, device=device, dtype=dtype),
            indexing='xy'
        )
        
        dirs = torch.stack([
            torch.zeros_like(i),
            torch.zeros_like(i),
            z_dir * torch.ones_like(i)
        ], dim=-1)
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
        
        origins = torch.stack([
            (i - K[0, 2]) / K[0, 0],
            -(j - K[1, 2]) / K[1, 1],
            torch.zeros_like(i)
        ], dim=-1)
        origins = torch.sum(origins[..., None, :] * c2w[:3, :3], dim=-1)
        rays_o = origins + c2w[:3, 3]
        
        return rays_o, rays_d
    
    def to_ndc_rays(self, rays_o: torch.Tensor, rays_d: torch.Tensor, near: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        
        H, W = rays_o.shape[:2]
        focal = self.intrinsics.fl_x  
        
        t = -(near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d
        
        o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1.0 + 2.0 * near / rays_o[..., 2]
        
        d0 = -1.0 / (W / (2.0 * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
        d1 = -1.0 / (H / (2.0 * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
        d2 = -2.0 * near / rays_o[..., 2]
        
        rays_o_ndc = torch.stack([o0, o1, o2], dim=-1)
        rays_d_ndc = torch.stack([d0, d1, d2], dim=-1)
        
        return rays_o_ndc, rays_d_ndc