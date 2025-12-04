import numpy as np 
import torch 
from typing import Optional

from .camera_extrinsics import CameraExtrinsics
from .camera_intrinsics import CameraIntrinsics

class Camera:
    def __init__(self, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics, 
                 name: Optional[str]=None, image_path: Optional[str]=None, device='cpu'):
        """Camera class with proper device management
        
        Args:
            intrinsics (CameraIntrinsics): Camera intrinsic parameters
            extrinsics (CameraExtrinsics): Camera extrinsic parameters (pose)
            name (Optional[str]): Camera name
            image_path (Optional[str]): Path to associated image
            device (str): Device to place tensors on ('cpu' or 'cuda')
        """
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.name = name
        self.image_path = image_path
        self.device = device
        
        # Store PyTorch versions on the correct device
        self.R = extrinsics.R.to(device)  # Keep as torch tensor
        self.t = extrinsics.t.to(device)  # Keep as torch tensor
        self.K = intrinsics.get_K().to(device)  # Ensure on correct device
        
        # Create Rt and P matrices as torch tensors
        self.Rt = torch.cat([self.R, self.t], dim=1)  # (3, 4) torch tensor
        self.P = self.K @ self.Rt  # (3, 4) Projection matrix
        
        # Keep NumPy versions for backward compatibility (if needed)
        self.R_np = self.R.cpu().numpy()
        self.t_np = self.t.cpu().numpy()
        self.K_np = self.K.cpu().numpy()
        self.Rt_np = self.Rt.cpu().numpy()
        self.P_np = self.P.cpu().numpy()
    
    def project(self, point3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D image coordinates (NumPy version)
        
        Args:
            point3d (np.ndarray): (N, 3) array of 3D points
            
        Returns:
            np.ndarray: (N, 2) array of 2D image coordinates
        """
        points_h = np.hstack([point3d, np.ones((point3d.shape[0], 1))])
        proj = (self.P_np @ points_h.T).T
        uv = proj[:, :2] / proj[:, 2:3]
        return uv
    
    def project_torch(self, point3d: torch.Tensor) -> torch.Tensor:
        """Project 3D points to 2D image coordinates (PyTorch version)
        
        Args:
            point3d (torch.Tensor): (N, 3) tensor of 3D points
            
        Returns:
            torch.Tensor: (N, 2) tensor of 2D image coordinates
        """
        point3d = point3d.to(self.device)
        ones = torch.ones((point3d.shape[0], 1), device=self.device, dtype=point3d.dtype)
        points_h = torch.cat([point3d, ones], dim=1)
        proj = (self.P @ points_h.T).T
        uv = proj[:, :2] / proj[:, 2:3]
        return uv
    
    def unproject(self, pixels: np.ndarray) -> np.ndarray:
        """Unproject 2D pixels to 3D rays in camera space (NumPy version)
        
        Args:
            pixels (np.ndarray): (N, 2) array of pixel coordinates
            
        Returns:
            np.ndarray: (N, 3) array of normalized ray directions
        """
        fx, fy = self.intrinsics.fl_x, self.intrinsics.fl_y
        cx, cy = self.intrinsics.cx, self.intrinsics.cy
        
        x = (pixels[:, 0] - cx) / fx
        y = (pixels[:, 1] - cy) / fy
        rays = np.stack([x, y, np.ones_like(x)], axis=-1)
        rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        return rays
    
    def unproject_torch(self, pixels: torch.Tensor) -> torch.Tensor:
        """Unproject 2D pixels to 3D rays in camera space (PyTorch version)
        
        Args:
            pixels (torch.Tensor): (N, 2) tensor of pixel coordinates
            
        Returns:
            torch.Tensor: (N, 3) tensor of normalized ray directions
        """
        pixels = pixels.to(self.device)
        fx, fy = self.intrinsics.fl_x, self.intrinsics.fl_y
        cx, cy = self.intrinsics.cx, self.intrinsics.cy
        
        x = (pixels[:, 0] - cx) / fx
        y = (pixels[:, 1] - cy) / fy
        ones = torch.ones_like(x)
        rays = torch.stack([x, y, ones], dim=-1)
        rays = rays / torch.norm(rays, dim=-1, keepdim=True)
        return rays
    
    def to_torch(self, device=None, dtype=torch.float32):
        """Convert camera matrices to torch tensors
        
        Args:
            device (str, optional): Device to place tensors on. If None, uses self.device
            dtype: Data type for tensors
            
        Returns:
            dict: Dictionary of camera matrices as torch tensors
        """
        if device is None:
            device = self.device
            
        return {
            "K": self.K.to(device=device, dtype=dtype),
            "R": self.R.to(device=device, dtype=dtype),
            "t": self.t.to(device=device, dtype=dtype),
            "Rt": self.Rt.to(device=device, dtype=dtype),
            "P": self.P.to(device=device, dtype=dtype)
        }
    
    def rays(self, H: int, W: int, device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate ray origins and directions for all pixels (FIXED VERSION)
        
        Args:
            H (int): Image height
            W (int): Image width
            device (str, optional): Device to place tensors on. If None, uses self.device
            dtype: Data type for tensors
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (origins, directions) both (H*W, 3)
        """
        if device is None:
            device = self.device
        
        # Create pixel grid on the correct device
        i, j = torch.meshgrid(
            torch.arange(W, device=device, dtype=dtype),
            torch.arange(H, device=device, dtype=dtype),
            indexing='xy'
        )
        pixels = torch.stack([i, j], dim=-1).reshape(-1, 2)  # (H*W, 2)
        
        d_cam = self.unproject_torch(pixels)  # (H*W, 3) in camera space
        
        # Get camera position
        pos = self.extrinsics.get_position().to(device=device, dtype=dtype)
        pos = pos.reshape(1, 3)
        
        # Transform to world space
        d_world = self.extrinsics.camera2world(d_cam) - pos  # (H*W, 3)
        d_world = d_world / torch.norm(d_world, dim=-1, keepdim=True)
        
        origin = pos.expand_as(d_world)  # Ray origins
        
        return origin, d_world
    
    def get_rays_perspective(self, H: int, W: int, device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate perspective projection rays (NeRF-style)
        
        Args:
            H (int): Image height
            W (int): Image width
            device (str, optional): Device to place tensors on. If None, uses self.device
            dtype: Data type for tensors
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (origins, directions) both (H, W, 3)
        """
        if device is None:
            device = self.device
            
        K = self.intrinsics.get_K().to(device=device, dtype=dtype)
        c2w = self.extrinsics.get_c2w().to(device=device, dtype=dtype)
        
        # Pixel grid
        i, j = torch.meshgrid(
            torch.linspace(0, W-1, W, device=device, dtype=dtype),
            torch.linspace(0, H-1, H, device=device, dtype=dtype),
            indexing='xy'
        )
        
        # Direction vectors in camera space
        dirs = torch.stack([
            (i - K[0, 2]) / K[0, 0],
            -(j - K[1, 2]) / K[1, 1],
            -torch.ones_like(i)
        ], dim=-1)  # (H, W, 3)
        
        # Transform to world space
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)  # (H, W, 3)
        rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)
        
        return rays_o, rays_d
    
    def get_rays_orthographic(self, H: int, W: int, device=None, dtype=torch.float32, z_dir=-1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate orthographic projection rays
        
        Args:
            H (int): Image height
            W (int): Image width
            device (str, optional): Device to place tensors on. If None, uses self.device
            dtype: Data type for tensors
            z_dir (float): Direction of rays in z (-1.0 for forward)
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (origins, directions) both (H, W, 3)
        """
        if device is None:
            device = self.device
            
        K = self.intrinsics.get_K().to(device=device, dtype=dtype)
        c2w = self.extrinsics.get_c2w().to(device=device, dtype=dtype)
        
        # Pixel grid
        i, j = torch.meshgrid(
            torch.linspace(0, W-1, W, device=device, dtype=dtype),
            torch.linspace(0, H-1, H, device=device, dtype=dtype),
            indexing='xy'
        )
        
        # Parallel rays direction
        dirs = torch.stack([
            torch.zeros_like(i),
            torch.zeros_like(i),
            z_dir * torch.ones_like(i)
        ], dim=-1)
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
        
        # Origins distributed across image plane
        origins = torch.stack([
            (i - K[0, 2]) / K[0, 0],
            -(j - K[1, 2]) / K[1, 1],
            torch.zeros_like(i)
        ], dim=-1)
        origins = torch.sum(origins[..., None, :] * c2w[:3, :3], dim=-1)
        rays_o = origins + c2w[:3, 3]
        
        return rays_o, rays_d
    
    def to_ndc_rays(self, rays_o: torch.Tensor, rays_d: torch.Tensor, near: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert rays to Normalized Device Coordinates (NDC)
        
        Args:
            rays_o (torch.Tensor): Ray origins (H, W, 3)
            rays_d (torch.Tensor): Ray directions (H, W, 3)
            near (float): Near plane distance
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (origins_ndc, directions_ndc)
        """
        H, W = rays_o.shape[:2]
        focal = self.intrinsics.fl_x  
        
        # Shift ray origins to near plane
        t = -(near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d
        
        # NDC transformation
        o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1.0 + 2.0 * near / rays_o[..., 2]
        
        d0 = -1.0 / (W / (2.0 * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
        d1 = -1.0 / (H / (2.0 * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
        d2 = -2.0 * near / rays_o[..., 2]
        
        rays_o_ndc = torch.stack([o0, o1, o2], dim=-1)
        rays_d_ndc = torch.stack([d0, d1, d2], dim=-1)
        
        return rays_o_ndc, rays_d_ndc
    
    def to(self, device):
        """Move camera to specified device
        
        Args:
            device: Target device ('cpu' or 'cuda')
            
        Returns:
            self for method chaining
        """
        self.device = device
        self.R = self.R.to(device)
        self.t = self.t.to(device)
        self.K = self.K.to(device)
        self.Rt = self.Rt.to(device)
        self.P = self.P.to(device)
        
        self.intrinsics.to(device)
        self.extrinsics.to(device)
        
        return self