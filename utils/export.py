import numpy as np 
import torch
from imageio
from pathlib import Path
from typing import Literal, Union, List 

class Exporter:
    @staticmethod
    def to_uint8(x: Union[np.ndarray, torch.Tensor])-> np.ndarray:
        """_summary_

        Args:
            x (Union[np.ndarray, torch.Tensor]): _description_

        Returns:
            np.ndarray: _description_
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        return x
    
    @staticmethod
    def save_image(image: Union[np.ndarray, torch.Tensor], path: Union[str, Path], save_npy: bool = False)-> None:
        """_summary_

        Args:
            image (Union[np.ndarray, torch.Tensor]): _description_
            path (Union[str, Path]): _description_
            save_npy (bool, optional): _description_. Defaults to False.
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
        images: List[Union[np.ndarray, torch.Tensor]], output_dir: Union[str, Path], prefix: str = "frame",
        save_npy: bool = False, verbose: bool = True
    )-> List[Path]:
        """_summary_

        Args:
            images (List[Union[np.ndarray, torch.Tensor]]): _description_
            output_dir (Union[str, Path]): _description_
            prefix (str, optional): _description_. Defaults to "frame".
            save_npy (bool, optional): _description_. Defaults to False.
            verbose (bool, optional): _description_. Defaults to True.

        Returns:
            List[Path]: _description_
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
                print(f"Saved image: {path}")
        
        if verbose:
            print(f"All images saved to {output_dir}")

        return saved_paths
    
    @staticmethod
    def save_video(
        images: List[Union[np.ndarray, torch.Tensor]], output_path: Union[str, Path], 
        fps: int = 30,quality: int = 8, verbose: bool = True)->Path:
        """_summary_

        Args:
            images (List[Union[np.ndarray, torch.Tensor]]): _description_
            output_path (Union[str, Path]): _description_
            fps (int, optional): _description_. Defaults to 30.
            quality (int, optional): _description_. Defaults to 8.
            verbose (bool, optional): _description_. Defaults to True.

        Returns:
            Path: _description_
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(images, torch.Tensor):
            images = [img for img in images]
        elif isinstance(images, list):
            images = np.stack([
                img.cpu().numpy() if isinstance(img, torch.Tensor) else img for img in images])
        
        images_uint8 = Exporter.to_uint8(images)
        imageio.mimwrite(path, images_uint8, fps=fps, quality=quality)
        
        if verbose:
            print(f"Saved video: {path}")
        
        return path
    
    @staticmethod
    def video2images(image_dir: Union[str, Path], output_dir: Union[str, Path], fps: int = 30,
                    quality: int = 8, pattern: str = "*.png")-> Path:
        """_summary_

        Args:
            image_dir (Union[str, Path]): _description_
            output_dir (Union[str, Path]): _description_
            fps (int, optional): _description_. Defaults to 30.
            quality (int, optional): _description_. Defaults to 8.
            pattern (str, optional): _description_. Defaults to "*.png".

        Returns:
            Path: _description_
        """
        if not image_paths:
            raise ValueError(f"No images found in {image_dir} with pattern {pattern}")
        
        images = [imageio.imread(path) for path in image_paths]

        return Exporter.save_video(images, output_path, fps=fps, quality=quality)
    

def export_images(images: List, output_dir: Union[str, Path], prefix: str = "frame"
                  save_npy: bool = False) -> List[Path]:
    """_summary_

    Args:
        images (List): _description_
        output_dir (Union[str, Path]): _description_
        prefix (_type_, optional): _description_. Defaults to "frame"save_npy:bool=False.

    Returns:
        List[Path]: _description_
    """
    return Exporter.save_images(images, output_dir, prefix=prefix, save_npy=save_npy)

def export_video(images: Union[List, np.ndarray, torch.Tensor],
                path: Union[str, Path], fps: int = 30, quality: int = 8) -> Path: 
    """_summary_

    Args:
        images (Union[List, np.ndarray, torch.Tensor]): _description_
        path (Union[str, Path]): _description_
        fps (int, optional): _description_. Defaults to 30.

    Returns:
        Path: _description_
    """
    return Exporter.save_video(images, path, fps=fps, quality=quality)