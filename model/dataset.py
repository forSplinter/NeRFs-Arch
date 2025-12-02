import os
import math
import numpy as np
import torch
import json
from typing import Optional, List, Tuple, Dict
import torch.utils.data
from camera.dataset_loader import DatasetLoader
import imageio.v3 as iio


class BaseNeRFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_path: str,
        rgb_paths: Optional[List[str]] = None,
        cam_id: bool = False,
        split: str = "train",
    ):
        """
        Base dataset class for NeRF.

        Args:
            json_path: Path to transforms.json
            rgb_paths: Optional list of paths to RGB images/numpy files
            cam_id: Whether to include camera IDs
            split: Dataset split ('train', 'val', 'test')
        """
        super().__init__()

        self.dataset_loader = DatasetLoader(json_path)
        self.cameras = self.dataset_loader.frames
        self.intrinsics = self.dataset_loader.intrinsics

        self.image_count = len(self.cameras)
        self.H = self.intrinsics.h
        self.W = self.intrinsics.w
        self.focal_x = self.intrinsics.fl_x
        self.focal_y = self.intrinsics.fl_y

        # Get near/far planes from JSON if available
        self.data = self.dataset_loader.data
        self.near = self.data.get("near", 0.1)
        self.far = self.data.get("far", 100.0)
        self.aabb_scale = self.data.get("aabb_scale", 16)

        self.rgb_paths = rgb_paths
        self.has_cam_id = cam_id
        self.split = split

        # Load RGB images if paths provided
        if self.rgb_paths is not None and len(self.rgb_paths) == len(self.cameras):
            rgbs_list = []
            for path in self.rgb_paths:
                if path.endswith(".npy"):
                    rgb = np.load(path)
                else:
                    # Load image file (you might need PIL or similar)
                    import imageio.v3 as iio

                    rgb = iio.imread(path)
                rgbs_list.append(torch.from_numpy(rgb).float())
            self.rgbs = torch.stack(rgbs_list)  # [N, H, W, 3]
        else:
            self.rgbs = None

        # Camera IDs for per-camera encoding
        if cam_id:
            self.cam_ids = torch.arange(self.image_count, dtype=torch.long)

    def num_images(self) -> int:
        """Return number of images/cameras in dataset."""
        return self.image_count

    def height_width(self) -> Tuple[int, int]:
        """Return image height and width."""
        return self.H, self.W

    def near_far(self) -> Tuple[float, float]:
        """Return near and far planes."""
        return self.near, self.far

    def radii(self) -> float:
        """Return radii for Mip-NeRF."""
        return 2.0 / max(self.H, self.W) * 2 / math.sqrt(12)

    def get_intrinsics(self, idx: int) -> Dict:
        """Get camera intrinsics for a specific camera."""
        camera = self.cameras[idx]
        return {
            "focal_x": self.focal_x,
            "focal_y": self.focal_y,
            "cx": self.intrinsics.cx,
            "cy": self.intrinsics.cy,
            "width": self.W,
            "height": self.H,
        }

    def get_camera_pose(self, idx: int) -> torch.Tensor:
        """Get camera pose (c2w matrix) for a specific camera."""
        return self.cameras[idx].extrinsics.get_c2w()

    def __len__(self) -> int:
        """Return dataset length."""
        if self.split == "train":
            return self.H * self.W * self.image_count
        else:
            return self.image_count

    def __getitem__(self, idx: int) -> Dict:
        """Get item from dataset."""
        raise NotImplementedError("Subclasses must implement __getitem__")


class RayNeRFDataset(BaseNeRFDataset):
    def __init__(
        self,
        json_path: str,
        rgb_paths: Optional[List[str]] = None,
        cam_id: bool = False,
        split: str = "train",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Ray dataset for NeRF that precomputes rays.

        Args:
            json_path: Path to transforms.json
            rgb_paths: Optional list of paths to RGB images
            cam_id: Whether to include camera IDs
            split: Dataset split ('train', 'val', 'test')
            device: Device for tensors
            dtype: Data type for tensors
        """
        super().__init__(json_path, rgb_paths, cam_id, split)

        self.device = device
        self.dtype = dtype

        # Precompute all rays
        self._precompute_rays()

    def _precompute_rays(self):
        """Precompute rays for all cameras."""
        rays_o_list = []
        rays_d_list = []

        for i, camera in enumerate(self.cameras):
            # Get rays from camera
            origin, direction = camera.rays(
                self.H, self.W, device=self.device, dtype=self.dtype
            )

            # Store as numpy for now, convert to tensor later
            rays_o_list.append(origin.cpu().numpy())
            rays_d_list.append(direction.cpu().numpy())

        # Stack all rays
        rays_o = np.stack(rays_o_list, axis=0)  # [N, H*W, 3]
        rays_d = np.stack(rays_d_list, axis=0)  # [N, H*W, 3]

        # Reshape to image dimensions
        rays_o = rays_o.reshape(self.image_count, self.H, self.W, 3)
        rays_d = rays_d.reshape(self.image_count, self.H, self.W, 3)

        # Convert to torch tensors
        self.rays_o = torch.from_numpy(rays_o).to(self.device, self.dtype)
        self.rays_d = torch.from_numpy(rays_d).to(self.device, self.dtype)

        # Stack rays_o and rays_d together
        self.all_rays = torch.stack(
            [self.rays_o, self.rays_d], dim=1
        )  # [N, 2, H, W, 3]

    def get_rays(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get rays for a specific camera."""
        return self.rays_o[idx], self.rays_d[idx]

    def __getitem__(self, idx: int) -> Dict:
        """Get item from dataset."""
        if self.split == "train":
            # For training: random pixel from random image
            img_idx = idx // (self.H * self.W)
            pix_idx = idx % (self.H * self.W)

            # Convert flat index to 2D coordinates
            y = pix_idx // self.W
            x = pix_idx % self.W

            # Get specific ray
            rays_o = self.rays_o[img_idx, y, x]  # [3]
            rays_d = self.rays_d[img_idx, y, x]  # [3]

            # Stack rays
            rays = torch.stack([rays_o, rays_d], dim=0)  # [2, 3]

            # Get target RGB if available
            target = None
            if self.rgbs is not None:
                target = self.rgbs[img_idx, y, x]  # [3]

            # Create sample dictionary
            sample = {"rays": rays, "target_s": target}

            # Add camera ID if requested
            if self.has_cam_id:
                sample["cam_id"] = self.cam_ids[img_idx]

            return sample

        else:
            # For validation/testing: return all rays for a camera
            img_idx = idx

            sample = {"rays": self.all_rays[img_idx]}  # [2, H, W, 3]

            # Add target RGB if available
            if self.rgbs is not None:
                sample["target_s"] = self.rgbs[img_idx]  # [H, W, 3]

            # Add camera ID if requested
            if self.has_cam_id:
                sample["cam_id"] = self.cam_ids[img_idx]

            return sample

    def get_full_batch(self, batch_size: int) -> Dict:
        """_summary_

        Args:
            batch_size (int): _description_

        Returns:
            Dict: _description_
        """
        indices = torch.randint(0, len(self), (batch_size,))
        batch = [self[int(i.item())] for i in indices]

        # Stack batch
        rays = torch.stack([item["rays"] for item in batch], dim=0)  # [B, 2, 3]

        # Get targets if available
        if self.rgbs is not None:
            targets = torch.stack([item["target_s"] for item in batch], dim=0)  # [B, 3]
        else:
            targets = None

        result = {"rays": rays}
        if targets is not None:
            result["target_s"] = targets

        # Add camera IDs if available
        if self.has_cam_id:
            cam_ids = torch.stack([item["cam_id"] for item in batch], dim=0)
            result["cam_id"] = cam_ids

        return result


# Helper function for creating datasets
def create_nerf_datasets(
    json_path: str,
    train_rgb_paths: Optional[List[str]] = None,
    val_rgb_paths: Optional[List[str]] = None,
    cam_id: bool = False,
    device: str = "cpu",
) -> Tuple[RayNeRFDataset, RayNeRFDataset]:
    """_summary_

    Args:
        json_path (str): _description_
        train_rgb_paths (Optional[List[str]], optional): _description_. Defaults to None.
        val_rgb_paths (Optional[List[str]], optional): _description_. Defaults to None.
        cam_id (bool, optional): _description_. Defaults to False.
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        Tuple[RayNeRFDataset, RayNeRFDataset]: _description_
    """
    train_dataset = RayNeRFDataset(
        json_path=json_path,
        rgb_paths=train_rgb_paths,
        cam_id=cam_id,
        split="train",
        device=device,
    )

    val_dataset = RayNeRFDataset(
        json_path=json_path,
        rgb_paths=val_rgb_paths,
        cam_id=cam_id,
        split="val",
        device=device,
    )

    return train_dataset, val_dataset
