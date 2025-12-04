import os
import math
import numpy as np
import torch
from typing import Optional, List, Tuple, Dict
import torch.utils.data
from utils.camera.dataset_loader import DatasetLoader
import imageio.v3 as iio


class BaseNeRFDataset(torch.utils.data.Dataset):
    def __init__(self, json_path: str, data_root: Optional[str] = None, cam_id: bool = False, split: str = "train", device: str = "cpu", dtype: torch.dtype = torch.float32):
        """_summary_

        Args:
            json_path (str): _description_
            data_root (Optional[str], optional): _description_. Defaults to None.
            cam_id (bool, optional): _description_. Defaults to False.
            split (str, optional): _description_. Defaults to "train".
            device (str, optional): _description_. Defaults to "cpu".
            dtype (torch.dtype, optional): _description_. Defaults to torch.float32.
        """
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.dataset_loader = DatasetLoader(json_path)
        self.cameras = self.dataset_loader.frames  
        self.intrinsics = self.dataset_loader.intrinsics  

        self.image_count = len(self.cameras)
        self.H = self.intrinsics.h
        self.W = self.intrinsics.w
        self.focal_x = self.intrinsics.fl_x
        self.focal_y = self.intrinsics.fl_y

        self.data = self.dataset_loader.data
        self.near = self.data.get("near", 0.1)
        self.far = self.data.get("far", 100.0)
        self.aabb_scale = self.data.get("aabb_scale", 16)

        self.data_root = data_root
        self.has_cam_id = cam_id
        self.split = split

        self.rgbs = self._load_images_from_json()

        if cam_id:
            self.cam_ids = torch.arange(self.image_count, dtype=torch.long)

    def _load_images_from_json(self):
        rgbs_list = []
        
        for i, camera in enumerate(self.cameras):
            if hasattr(camera, 'image_path') and camera.image_path:
                img_path = camera.image_path
                
                if self.data_root:
                    base_path = os.path.dirname(img_path)
                    filename = os.path.basename(img_path)
                    img_path = os.path.join(self.data_root, filename)
                
                img_path = os.path.normpath(img_path)
                
                try:
                    if img_path.endswith('.npy'):
                        rgb = np.load(img_path)
                    else:
                        rgb = iio.imread(img_path)
                    
                    if rgb.dtype == np.uint8:
                        rgb = rgb.astype(np.float32) / 255.0
                    
                    if len(rgb.shape) == 2:
                        rgb = np.stack([rgb, rgb, rgb], axis=-1)
                    elif rgb.shape[2] == 4:
                        rgb = rgb[:, :, :3]
                    
                    rgbs_list.append(torch.from_numpy(rgb).float())
                    
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}")
                    dummy_rgb = torch.zeros((self.H, self.W, 3), dtype=torch.float32)
                    rgbs_list.append(dummy_rgb)
            else:
                dummy_rgb = torch.zeros((self.H, self.W, 3), dtype=torch.float32)
                rgbs_list.append(dummy_rgb)
        
        if rgbs_list:
            return torch.stack(rgbs_list)
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
        camera = self.cameras[idx]
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
    def __init__(self,json_path: str,data_root: Optional[str] = None,cam_id: bool = False,split: str = "train",device: str = "cpu",dtype: torch.dtype = torch.float32,):
        super().__init__(json_path, data_root, cam_id, split)
        self.device = device
        self.dtype = dtype
        self._precompute_rays()

    def _precompute_rays(self):
        rays_o_list = []
        rays_d_list = []

        for i, camera in enumerate(self.cameras):
            origin, direction = camera.rays(
                self.H, self.W, device=self.device, dtype=self.dtype
            )

            rays_o_list.append(origin.cpu().numpy())
            rays_d_list.append(direction.cpu().numpy())

        rays_o = np.stack(rays_o_list, axis=0)
        rays_d = np.stack(rays_d_list, axis=0)

        rays_o = rays_o.reshape(self.image_count, self.H, self.W, 3)
        rays_d = rays_d.reshape(self.image_count, self.H, self.W, 3)

        self.rays_o = torch.from_numpy(rays_o).to(self.device, self.dtype)
        self.rays_d = torch.from_numpy(rays_d).to(self.device, self.dtype)

        self.all_rays = torch.stack([self.rays_o, self.rays_d], dim=1)

    def get_rays(self, idx: int)-> Tuple[torch.Tensor, torch.Tensor]:
        return self.rays_o[idx], self.rays_d[idx]

    def __getitem__(self, idx: int)-> dict:
        if self.split == "train":
            img_idx = idx // (self.H * self.W)
            pix_idx = idx % (self.H * self.W)

            y = pix_idx // self.W
            x = pix_idx % self.W

            rays_o = self.rays_o[img_idx, y, x]
            rays_d = self.rays_d[img_idx, y, x]

            rays = torch.stack([rays_o, rays_d], dim=0)

            target = None
            if self.rgbs is not None:
                target = self.rgbs[img_idx, y, x]

            sample = {"rays": rays, "target_s": target}

            if self.has_cam_id:
                sample["cam_id"] = self.cam_ids[img_idx]

            return sample

        else:
            img_idx = idx

            sample = {"rays": self.all_rays[img_idx]}

            if self.rgbs is not None:
                sample["target_s"] = self.rgbs[img_idx]

            if self.has_cam_id:
                sample["cam_id"] = self.cam_ids[img_idx]

            return sample

    def get_full_batch(self, batch_size: int)-> dict:
        indices = torch.randint(0, len(self), (batch_size,))
        batch = [self[int(i.item())] for i in indices]

        rays = torch.stack([item["rays"] for item in batch], dim=0)

        if self.rgbs is not None:
            targets = torch.stack([item["target_s"] for item in batch], dim=0)
        else:
            targets = None

        result = {"rays": rays}
        if targets is not None:
            result["target_s"] = targets

        if self.has_cam_id:
            cam_ids = torch.stack([item["cam_id"] for item in batch], dim=0)
            result["cam_id"] = cam_ids

        return result


def create_nerf_datasets(json_path: str, train_data_root: Optional[str] = None, val_data_root: Optional[str] = None,cam_id: bool = False,device: str = "cpu")-> Tuple[RayNeRFDataset, RayNeRFDataset]:
    train_dataset = RayNeRFDataset(
        json_path=json_path,
        data_root=train_data_root,
        cam_id=cam_id,
        split="train",
        device=device,
    )

    val_dataset = RayNeRFDataset(
        json_path=json_path,
        data_root=val_data_root,
        cam_id=cam_id,
        split="val",
        device=device,
    )

    return train_dataset, val_dataset