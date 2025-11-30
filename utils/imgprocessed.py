import os 
from PIL import Image, ImageOps
import numpy as np 

class ImageProcessor:
    @staticmethod
    def resize_image(image_path: str, output_path: str, max_size: int, quality: int = 95):
        """Resize an image to fit within max_size while maintaining aspect ratio.

        Args:
            image_path (str): Path to the input image.
            output_path (str): Path to save the resized image.
            max_size (int): Maximum size for the longest side of the image.
            quality (int, optional): Quality for saving JPEG images. Defaults to 95.
        """
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size
            ratio = width / height
            
            if width > height:
                new_width = min(width, max_size)
                new_height = int(new_width / ratio)
            else:
                new_height = min(height, max_size)
                new_width = int(new_height * ratio)
            
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                img_resized.save(output_path, quality=quality, optimize=True)
            else:
                img_resized.save(output_path, optimize=True)
    
    @staticmethod
    def convert_to_rgba(image_path: str, output_path: str):
        """Convert an image to RGBA format.

        Args:
            image_path (str): Path to the input image.
            output_path (str): Path to save the converted image.
        """
        with Image.open(image_path) as img:
            img_rgba = img.convert("RGBA")
            img_rgba.save(output_path, optimize=True)
    
    @staticmethod
    def get_image_stats(image_path: str) -> dict:
        """Get basic statistics of an image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: Dictionary containing image format, size, and mode.
        """
        with Image.open(image_path) as img:
            return {
                "format": img.format,
                "size": img.size,
                "mode": img.mode
            }