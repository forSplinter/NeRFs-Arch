"""
colmap_utils.py - Utilitaires pour la gestion des données COLMAP
"""

import os
import shutil
import json
import struct
import numpy as np
from collections import namedtuple
from typing import Dict, List, Optional

Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])


class ColmapFileIO:
    """Classe pour la lecture des fichiers COLMAP"""
    
    @staticmethod
    def read_cameras_binary(path_to_model_file: str) -> Dict[int, Camera]:
        cameras = {}
        with open(path_to_model_file, "rb") as fid:
            num_cameras = struct.unpack("Q", fid.read(8))[0]
            for _ in range(num_cameras):
                camera_id = struct.unpack("I", fid.read(4))[0]
                model_id = struct.unpack("i", fid.read(4))[0]
                width = struct.unpack("Q", fid.read(8))[0]
                height = struct.unpack("Q", fid.read(8))[0]
                num_params = struct.unpack("Q", fid.read(8))[0]
                params = struct.unpack("d" * num_params, fid.read(8 * num_params))
                cameras[camera_id] = Camera(id=camera_id, model=model_id, width=width, height=height, params=params)
        return cameras
    
    @staticmethod
    def read_images_binary(path_to_model_file: str) -> Dict[int, Image]:
        images = {}
        with open(path_to_model_file, "rb") as fid:
            num_reg_images = struct.unpack("Q", fid.read(8))[0]
            for _ in range(num_reg_images):
                image_id = struct.unpack("I", fid.read(4))[0]
                qvec = struct.unpack("dddd", fid.read(32))
                tvec = struct.unpack("ddd", fid.read(24))
                camera_id = struct.unpack("I", fid.read(4))[0]
                image_name = ""
                current_char = struct.unpack("c", fid.read(1))[0]
                while current_char != b"\x00":
                    image_name += current_char.decode("utf-8")
                    current_char = struct.unpack("c", fid.read(1))[0]
                num_points2D = struct.unpack("Q", fid.read(8))[0]
                xys = struct.unpack("d" * 2 * num_points2D, fid.read(16 * num_points2D))
                point3D_ids = struct.unpack("i" * num_points2D, fid.read(4 * num_points2D))
                images[image_id] = Image(
                    id=image_id, qvec=np.array(qvec), tvec=np.array(tvec),
                    camera_id=camera_id, name=image_name,
                    xys=np.array(xys).reshape(-1, 2), point3D_ids=np.array(point3D_ids)
                )
        return images


class ColmapUtils:
    def __init__(self, dataset_root: str):
        self.dataset_root = dataset_root
        self.processed_path = os.path.join(dataset_root, "processed")
        self.colmap_dir = os.path.join(dataset_root, "colmap")
    
    def setup_colmap(self, lot_name: str, size: int) -> str:
        base_dir = os.path.join(self.colmap_dir, lot_name, f"{size}px")
        directories = ["images", "sparse", "dense", "output"]
        for dir_name in directories:
            os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)
        return base_dir
    
    def copy_mask(self, lot_name: str, size: int) -> bool:
        mask_src = os.path.join(self.processed_path, "masks", lot_name, f"{size}px")
        colmap_base = self.setup_colmap(lot_name, size)
        colmap_img = os.path.join(colmap_base, "images")

        if not os.path.exists(mask_src):
            return False
        
        png_files = [f for f in os.listdir(mask_src) if f.lower().endswith('.png')]
        for png_file in png_files:
            src_path = os.path.join(mask_src, png_file)
            dst_path = os.path.join(colmap_img, png_file)
            shutil.copy2(src_path, dst_path)
        
        return True
    
    def create_colmap_project(self, lot_name: str, size: int) -> str:
        colmap_base = os.path.join(self.colmap_dir, lot_name, f"{size}px")
        project_config = {
            "database_path": os.path.join(colmap_base, "database.db"),
            "image_path": os.path.join(colmap_base, "images"),
            "mask_path": os.path.join(colmap_base, "images")
        }
        project_file = os.path.join(colmap_base, "project.ini")
        with open(project_file, 'w') as f:
            f.write("[database]\n")
            f.write(f"database_path = {project_config['database_path']}\n\n")

            f.write("[images]\n")
            f.write(f"image_path = {project_config['image_path']}\n")
            f.write(f"mask_path = {project_config['mask_path']}\n\n")

            f.write("[sparse]\n")
            f.write(f"sparse_path = {os.path.join(colmap_base, 'sparse')}\n\n")

            f.write("[dense]\n")
            f.write(f"dense_path = {os.path.join(colmap_base, 'dense')}\n")
            
        return project_file
    
    def get_lots(self) -> List[str]:
        mask_dir = os.path.join(self.processed_path, "masks")
        if os.path.exists(mask_dir):
            return [
                d for d in os.listdir(mask_dir) 
                if os.path.isdir(os.path.join(mask_dir, d)) 
                and d.startswith("lot")
            ]
        return []
    
    def get_sizes(self, lot_name: str) -> List[int]:
        lot_dir = os.path.join(self.processed_path, "masks", lot_name)
        if os.path.exists(lot_dir):
            sizes = []
            for item in os.listdir(lot_dir):
                item_path = os.path.join(lot_dir, item)
                if os.path.isdir(item_path) and item.endswith("px"):
                    try: 
                        size = int(item.replace("px", ""))
                        sizes.append(size)
                    except ValueError:
                        continue
            return sorted(sizes)
        return []
    
    def save_metadata(self, lot_name: str, size: int, metadata: dict) -> str:
        colmap_base = self.setup_colmap(lot_name, size)
        metadata_path = os.path.join(colmap_base, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        return metadata_path
    
    def get_colmap_path(self, lot_name: str, size: int) -> str:
        return os.path.join(self.colmap_dir, lot_name, f"{size}px")


# Fonctions globales pour compatibilité
def read_cameras_binary(path_to_model_file: str) -> Dict[int, Camera]:
    return ColmapFileIO.read_cameras_binary(path_to_model_file)

def read_images_binary(path_to_model_file: str) -> Dict[int, Image]:
    return ColmapFileIO.read_images_binary(path_to_model_file)