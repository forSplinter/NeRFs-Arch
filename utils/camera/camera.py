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