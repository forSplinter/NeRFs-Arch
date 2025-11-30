import os 
import shutil
import subprocess
from utils.colmap_utils import ColmapUtils

class ColmapProcess:
    def __init__(self, dataset_root: str):
        self.dataset_root = dataset_root
        self.utils = ColmapUtils(dataset_root=dataset_root)
    
    def run_colmap_local(self, lot_name: str, size: int):
        if not self.utils.copy_mask(lot_name, size):
            return False
        
        colmap_base = os.path.join(self.utils.colmap_dir, lot_name, f"{size}px")

        commands = [
            f'colmap feature_extractor\
                --database_path "{os.path.join(colmap_base, "database.db")}"\
                --image_path "{os.path.join(colmap_base, "images")}"\
                --ImageReader.mask_path "{os.path.join(colmap_base, "images")}"\
                --ImageReader.single_camera 1',
                
            f'colmap exhaustive_matcher\
                --database_path "{os.path.join(colmap_base, "database.db")}"',
            
            f'colmap mapper\
                --database_path "{os.path.join(colmap_base, "database.db")}"\
                --image_path "{os.path.join(colmap_base, "images")}"\
                --output_path "{os.path.join(colmap_base, "sparse")}"',
            
            f'colmap image_undistorter\
                --image_path "{os.path.join(colmap_base, "images")}"\
                --input_path "{os.path.join(colmap_base, "sparse", "0")}"\
                --output_path "{os.path.join(colmap_base, "dense")}"',
            
            f'colmap patch_match_stereo\
                --workspace_path "{os.path.join(colmap_base, "dense")}"\
                --PatchMatchStereo.geom_consistency true',
            f'colmap stereo_fusion\
                --workspace_path "{os.path.join(colmap_base, "dense")}"\
                --output_path "{os.path.join(colmap_base, "dense", "fused.ply")}"'
        ]
        
        for i, cmd in enumerate(commands):
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Command {i+1} failed with error: {result.stderr}")
                    return False
            except Exception as e:
                print(f"An error occurred while executing command {i+1}: {str(e)}")
                return False
        
        self._organize_output(lot_name, size)
        print(f"COLMAP processing completed for {lot_name} at {size}px")
        return True

    def prepare_cloud_data(self, lot_name: str, size: int):
        if not self.utils.copy_mask(lot_name, size):
            return False
        colmap_base = os.path.join(self.utils.colmap_dir, lot_name, f"{size}px")
        return colmap_base

    def _organize_output(self, lot_name: str, size: int):
        colmap_base = os.path.join(self.utils.colmap_dir, lot_name, f"{size}px")
        output_dir = os.path.join(colmap_base, "output")

        important_files = [
            os.path.join(colmap_base, "dense", "fused.ply"),
            os.path.join(colmap_base, "sparse", "0", "cameras.txt"),
            os.path.join(colmap_base, "sparse", "0", "images.txt"),
            os.path.join(colmap_base, "sparse", "0", "points3D.txt")
        ]

        for file_path in important_files:
            if os.path.exists(file_path):
                shutil.copy2(file_path, output_dir)
        
        print(f"Organized COLMAP output files to {output_dir}")