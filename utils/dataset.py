import os
import math
import numpy as np
import torch
from typing import Optional, List, Tuple, Dict
import torch.utils.data
from utils.camera.dataset_loader import DatasetLoader
import imageio.v3 as iio


class BaseNeRFDataset(torch.utils.data.Dataset):
    def __init__(self, json_path: str, data_root: Optional[str] = None, cam_id: bool = False, split: str = "train", device: str = "cpu", dtype: torch.dtype = torch.float32,
                 near: Optional[float] = None, far: Optional[float] = None):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.json_path = json_path
        self.json_dir = os.path.dirname(os.path.abspath(json_path))
        
        self.dataset_loader = DatasetLoader(json_path)
        self.cameras = self.dataset_loader.frames  
        self.intrinsics = self.dataset_loader.intrinsics  

        self.image_count = len(self.cameras)
        self.H = self.intrinsics.h
        self.W = self.intrinsics.w
        self.focal_x = self.intrinsics.fl_x
        self.focal_y = self.intrinsics.fl_y

        self.data = self.dataset_loader.data
        if near is not None and far is not None:
            self.near = near
            self.far = far
        else:
            self.near, self.far = self.auto_compute_near_far(self.dataset_loader)

        self.aabb_scale = self.data.get("aabb_scale", 16)

        self.data_root = data_root if data_root else self.json_dir
        self.has_cam_id = cam_id
        self.split = split

        self.rgbs = self._load_images_from_json()
        
        if self.rgbs is not None:
            print(f"Loaded {self.rgbs.shape[0]} images, shape: {self.rgbs.shape}")
            print(f"RGB range: [{self.rgbs.min():.3f}, {self.rgbs.max():.3f}]")
        else:
            print("WARNING: No images loaded!")

        if cam_id:
            self.cam_ids = torch.arange(self.image_count, dtype=torch.long)
    def auto_compute_near_far(self, dataset_loader, percentile_low=0.1, percentile_high=99.9, factor_near=0.9, factor_far=1.1):
        camera_positions = []
        for cam in dataset_loader.frames:
        # récupère position de caméra via ton Camera/Extrinsics
            pos = cam.extrinsics.get_position()  # shape (3,)
            if isinstance(pos, torch.Tensor):
                pos = pos.cpu().numpy()
            camera_positions.append(pos)
        camera_positions = np.stack(camera_positions) 
        
        if "points3D" in dataset_loader.data:
            points3D = np.array(dataset_loader.data["points3D"])  # [N_pts, 3]
            dists = np.linalg.norm(points3D[None, :, :] - camera_positions[:, None, :], axis=-1)
            dists = dists.flatten()
        else:
            dists = np.linalg.norm(camera_positions, axis=-1)
        
        near = np.percentile(dists, percentile_low) * factor_near
        far = np.percentile(dists, percentile_high) * factor_far
        return float(near), float(far)
    def _load_images_from_json(self):
        rgbs_list = []
        
        for i, camera in enumerate(self.cameras):
            if hasattr(camera, 'image_path') and camera.image_path:
                img_path = camera.image_path
                
                if not os.path.isabs(img_path):
                    img_path = os.path.join(self.json_dir, img_path)
                
                img_path = os.path.normpath(img_path)
                
                if i == 0:
                    print(f"Loading first image from: {img_path}")
                    print(f"File exists: {os.path.exists(img_path)}")
                
                try:
                    if img_path.endswith('.npy'):
                        rgb = np.load(img_path)
                    else:
                        rgb = iio.imread(img_path)
                    
                    if rgb.dtype == np.uint8:
                        rgb = rgb.astype(np.float32) / 255.0
                    elif rgb.dtype == np.uint16:
                        rgb = rgb.astype(np.float32) / 65535.0
                    
                    if len(rgb.shape) == 2:
                        rgb = np.stack([rgb, rgb, rgb], axis=-1)
                    elif rgb.shape[2] == 4:
                        rgb = rgb[:, :, :3]
                    
                    if i == 0:
                        print(f"First image shape: {rgb.shape}, range: [{rgb.min():.3f}, {rgb.max():.3f}]")
                    
                    rgbs_list.append(torch.from_numpy(rgb).float())
                    
                except Exception as e:
                    print(f"ERROR: Could not load image {img_path}: {e}")
                    dummy_rgb = torch.zeros((self.H, self.W, 3), dtype=torch.float32)
                    rgbs_list.append(dummy_rgb)
            else:
                print(f"WARNING: Camera {i} has no image_path")
                dummy_rgb = torch.zeros((self.H, self.W, 3), dtype=torch.float32)
                rgbs_list.append(dummy_rgb)
        
        if rgbs_list:
            stacked = torch.stack(rgbs_list)
            return stacked
        return None

    def num_images(self):
        return self.image_count
    
    def height_width(self):
        return self.H, self.W

    def near_far(self):
        return self.near, self.far

    def radii(self):
        return 2.0 / max(self.H, self.W) * 2 / math.sqrt(12)

    def get_intrinsics(self, idx: int):
        return {
            "focal_x": self.focal_x,
            "focal_y": self.focal_y,
            "cx": self.intrinsics.cx,
            "cy": self.intrinsics.cy,
            "width": self.W,
            "height": self.H,
        }

    def get_camera_pose(self, idx: int):
        return self.cameras[idx].extrinsics.get_c2w()

    def __len__(self):
        if self.split == "train":
            return self.H * self.W * self.image_count
        else:
            return self.image_count

    def __getitem__(self, idx: int):
        raise NotImplementedError("Subclasses must implement __getitem__")


class RayNeRFDataset(BaseNeRFDataset):
    def __init__(self, json_path: str, data_root: Optional[str] = None, cam_id: bool = False, split: str = "train", device: str = "cpu", dtype: torch.dtype = torch.float32,
                 near: Optional[float] = None, far: Optional[float] = None):
        # Force CPU pour le stockage, on transfère au GPU par batch
        self.target_device = device
        super().__init__(json_path, data_root, cam_id, split, 'cpu', dtype, near, far)
        self._precompute_rays()

    def _precompute_rays(self):
        rays_o_list = []
        rays_d_list = []

        for i, camera in enumerate(self.cameras):
            origin, direction = camera.rays(
                self.H, self.W, device='cpu', dtype=self.dtype
            )
            rays_o_list.append(origin.numpy())
            rays_d_list.append(direction.numpy())

        rays_o = np.stack(rays_o_list, axis=0)
        rays_d = np.stack(rays_d_list, axis=0)

        rays_o = rays_o.reshape(self.image_count, self.H, self.W, 3)
        rays_d = rays_d.reshape(self.image_count, self.H, self.W, 3)

        # Stocke sur CPU
        self.rays_o = torch.from_numpy(rays_o).to(dtype=self.dtype)
        self.rays_d = torch.from_numpy(rays_d).to(dtype=self.dtype)
        self.all_rays = torch.stack([self.rays_o, self.rays_d], dim=1)

    def get_full_batch(self, batch_size: int) -> dict:
        indices = torch.randint(0, len(self), (batch_size,))
        
        if self.split == "train":
            img_indices = indices // (self.H * self.W)
            pix_indices = indices % (self.H * self.W)
            y_coords = pix_indices // self.W
            x_coords = pix_indices % self.W
            
            rays_o = self.rays_o[img_indices, y_coords, x_coords]
            rays_d = self.rays_d[img_indices, y_coords, x_coords]
            rays = torch.stack([rays_o, rays_d], dim=1)
            
            if self.rgbs is not None:
                targets = self.rgbs[img_indices, y_coords, x_coords]
            else:
                targets = torch.zeros((batch_size, 3), dtype=self.dtype)
            
            # Transfère au GPU ici
            result = {
                "rays": rays.to(self.target_device),
                "target_s": targets.to(self.target_device)
            }
            
            if self.has_cam_id:
                result["cam_id"] = img_indices.to(self.target_device)
            
            return result
        else:
            batch = [self[int(i.item())] for i in indices]
            rays = torch.stack([item["rays"] for item in batch], dim=0)
            targets = torch.stack([item["target_s"] for item in batch], dim=0)
            
            result = {
                "rays": rays.to(self.target_device),
                "target_s": targets.to(self.target_device)
            }
            
            if self.has_cam_id:
                cam_ids = torch.stack([item["cam_id"] for item in batch], dim=0)
                result["cam_id"] = cam_ids.to(self.target_device)
            
            return result

def create_nerf_datasets(json_path: str, train_data_root: Optional[str] = None, val_data_root: Optional[str] = None, cam_id: bool = False, device: str = "cpu",
                         near: Optional[float] = None, far: Optional[float] = None) -> Tuple[RayNeRFDataset, RayNeRFDataset]:
    train_dataset = RayNeRFDataset(
        json_path=json_path,
        data_root=train_data_root,
        cam_id=cam_id,
        split="train",
        device=device,
        near=near,
        far=far,
    )

    val_dataset = RayNeRFDataset(
        json_path=json_path,
        data_root=val_data_root,
        cam_id=cam_id,
        split="val",
        device=device,
        near=near,
        far=far,
    )

    return train_dataset, val_dataset