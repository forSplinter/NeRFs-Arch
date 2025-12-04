# utils/export.py

import numpy as np 
import torch
import imageio
from pathlib import Path
from typing import Sequence, Union, List 


class Exporter:
    @staticmethod
    def to_uint8(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Convert float array to uint8 [0, 255]
        
        Args:
            x: Image in float format
        
        Returns:
            Image in uint8 format
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        return x
    
    @staticmethod
    def save_image(
        image: Union[np.ndarray, torch.Tensor], 
        path: Union[str, Path], 
        save_npy: bool = False
    ) -> Path:
        """
        Save a single image
        
        Args:
            image: Image to save
            path: Output path
            save_npy: Also save as .npy
        
        Returns:
            Path to saved image
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        image_uint8 = Exporter.to_uint8(image)
        imageio.imwrite(path, image_uint8)
        
        if save_npy:
            npy_path = path.with_suffix('.npy')
            np.save(npy_path, image)
        
        return path 
    
    @staticmethod
    def save_images(
        images: List[Union[np.ndarray, torch.Tensor]], 
        output_dir: Union[str, Path], 
        prefix: str = "frame",
        save_npy: bool = False, 
        verbose: bool = True
    ) -> List[Path]:
        """
        Save multiple images
        
        Args:
            images: List of images
            output_dir: Output directory
            prefix: Filename prefix
            save_npy: Also save as .npy
            verbose: Print progress
        
        Returns:
            List of saved paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, img in enumerate(images):
            filename = f"{prefix}_{i:04d}.png"
            path = output_dir / filename
            Exporter.save_image(img, path, save_npy=save_npy)
            saved_paths.append(path)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Saved {i + 1}/{len(images)} images")
        
        if verbose:
            print(f"✅ All images saved to {output_dir}")

        return saved_paths
    
    @staticmethod
    def save_video(
        images: Union[List[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor],
        output_path: Union[str, Path], 
        fps: int = 30,
        quality: int = 8, 
        verbose: bool = True
    ) -> Path:
        """
        Save video from images
        
        Args:
            images: List of images or stacked array
            output_path: Output video path
            fps: Frames per second
            quality: Video quality [1-10]
            verbose: Print info
        
        Returns:
            Path to saved video
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy array
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        elif isinstance(images, list):
            images = np.stack([
                img.cpu().numpy() if isinstance(img, torch.Tensor) else img 
                for img in images
            ])
        
         # Convert to uint8
        images_uint8 = Exporter.to_uint8(images)
        
        # Save video - convert array to list of frames
        imageio.mimwrite(path, [frame for frame in images_uint8], fps=fps, quality=quality)
        
        if verbose:
            print(f"✅ Saved video: {path} ({len(images)} frames @ {fps} fps)")
        
        return path
    
    @staticmethod
    def images_to_video(
        image_dir: Union[str, Path],
        output_path: Union[str, Path], 
        fps: int = 30,
        quality: int = 8, 
        pattern: str = "*.png"
    ) -> Path:
        """
        Create video from image directory
        
        Args:
            image_dir: Directory containing images
            output_path: Output video path
            fps: Frames per second
            quality: Video quality
            pattern: Glob pattern for images
        
        Returns:
            Path to saved video
        """
        image_dir = Path(image_dir)
        image_paths = sorted(image_dir.glob(pattern))
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir} with pattern {pattern}")
        
        images = [np.array(imageio.imread(path)) for path in image_paths]
        
        return Exporter.save_video(images, output_path, fps=fps, quality=quality)


def to8b(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert to uint8"""
    return Exporter.to_uint8(x)

def export_images(images: List, output_dir: Union[str, Path], prefix: str = "frame",save_npy: bool = False) -> List[Path]:
    """Export multiple images"""
    return Exporter.save_images(images, output_dir, prefix=prefix, save_npy=save_npy)


def export_video(images: Union[List, np.ndarray, torch.Tensor],path: Union[str, Path], fps: int = 30, quality: int = 8) -> Path:
    """Export video"""
    return Exporter.save_video(images, path, fps=fps, quality=quality)