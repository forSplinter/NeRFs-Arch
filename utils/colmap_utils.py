import os 
import shutil 
import json 
from pathlib import Path 

class ColmapUtils:
    def __init__(self, dataset_root: str):
        self.dataset_root = dataset_root
        self.processed_path = os.path.join(dataset_root, "processed")
        self.colmap_dir = os.path.join(dataset_root, "colmap")
    
    def setup_colmap(self, lot_name: str, size: int):
        """_summary_

        Args:
            lot_name (str): _description_
            size (int): _description_
        """
        base_dir = os.path.join(self.colmap_dir, lot_name, f"{size}px")
        directories = ["images", "sparse", "dense"]
        for dir_name in directories:
            os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)
        
        return base_dir
    
    def copy_mask(self, lot_name: str, size: int):
        """_summary_

        Args:
            lot_name (str): _description_
            size (int): _description_
        """
        mask_src = os.path.join(self.processed_path, "masks", lot_name, f"{size}px")
        colmap_base = self.setup_colmap(lot_name, size)
        colmap_img = os.path.join(colmap_base, "images")

        if not os.path.exists(mask_src):
            return False
        
        png_files =[f for f in os.listdir(mask_src) if f.lower().endswith('.png')]
        for png_file in png_files:
            src_path = os.path.join(mask_src, png_file)
            dst_path = os.path.join(colmap_img, png_file)
            shutil.copy2(src_path, dst_path)
        
        print(f"Copied {len(png_files)} mask files to {colmap_img}")
        return True
    
    def create_colmap(self, lot_name: str, size: int):
        """_summary_

        Args:
            lot_name (str): _description_
            size (int): _description_
        """
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
            f.write(f"image_path = {project_config['image_path']}\n\n")
            f.write(f"mask_path = {project_config['mask_path']}\n")

            f.write("[sparse]\n")
            f.write(f"sparse_path = {os.path.join(colmap_base, 'sparse')}\n\n")

            f.write("[dense]\n")
            f.write(f"dense_path = {os.path.join(colmap_base, 'dense')}\n")
            
        print(f"Project file {project_file}")

        return project_file
    
    def get_lots(self):
        mask_dir = os.path.join(self.processed_path, "masks")
        if os.path.exists(mask_dir):
            return [
                d for d in os.listdir(mask_dir) 
                if os.path.isdir(os.path.join(mask_dir, d)) 
                and d.startswith
            ]
        return []
    
    def get_sizes(self, lot_name: str):
        lot_dir = os.path.join(self.processed_path, "masks", lot_name)
        if os.path.exists(lot_dir):
            sizes = []
            for item in os.listdir(lot_dir):
                if item in os.listdir(lot_dir) and os.path.isdir(os.path.join(lot_dir, item)):
                    try: 
                        size = int(item.replace("px", ""))
                        sizes.append(size)
                    except ValueError:
                        continue
            return sorted(sizes)
        return []
    
    def save_metadata(self, lot_name: str, size: int, metadata: dict):
        """_summary_

        Args:
            lot_name (str): _description_
            size (int): _description_
            metadata (dict): _description_
        """
        colmap_base = self.setup_colmap(lot_name, size)
        metadata_path = os.path.join(colmap_base, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Saved metadata to {metadata_path}")
    