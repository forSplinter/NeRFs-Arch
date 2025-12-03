import os 
import shutil
import subprocess
from utils.colmap.colmap_utils import ColmapUtils
import platform
import sys

class ColmapProcess:
    def __init__(self, dataset_root: str):
        self.dataset_root = dataset_root
        self.utils = ColmapUtils(dataset_root=dataset_root)
        self.is_macos = platform.system() == 'Darwin'
    
    def run_colmap_local(self, lot_name: str, size: int, skip_dense: bool = True):
        if not self.utils.copy_mask(lot_name, size):
            return False
        
        colmap_base = os.path.join(self.utils.colmap_dir, lot_name, f"{size}px")
        
        cmd = [
            f'colmap feature_extractor\
                --database_path "{os.path.join(colmap_base, "database.db")}"\
                --image_path "{os.path.join(colmap_base, "images")}"\
                --ImageReader.mask_path "{os.path.join(colmap_base, "images")}"\
                --ImageReader.single_camera 1\
                --FeatureExtraction.gpu_index -1\
                --FeatureExtraction.num_threads {self._get_num_threads()}\
                --SiftExtraction.max_image_size 1600\
                --SiftExtraction.max_num_features 4096',
                
            # exhaustive_matcher avec ses propres flags
            f'colmap exhaustive_matcher\
                --database_path "{os.path.join(colmap_base, "database.db")}"\
                --FeatureMatching.use_gpu 0\
                --FeatureMatching.num_threads {self._get_num_threads()}',
            
            # mapper sans flags inutiles
            f'colmap mapper\
                --database_path "{os.path.join(colmap_base, "database.db")}"\
                --image_path "{os.path.join(colmap_base, "images")}"\
                --output_path "{os.path.join(colmap_base, "sparse")}"',
        ]
        
        if not skip_dense:
            cmd.extend([
                f'colmap image_undistorter\
                    --image_path "{os.path.join(colmap_base, "images")}"\
                    --input_path "{os.path.join(colmap_base, "sparse", "0")}"\
                    --output_path "{os.path.join(colmap_base, "dense")}"',
                
                f'colmap patch_match_stereo\
                    --workspace_path "{os.path.join(colmap_base, "dense")}"\
                    --PatchMatchStereo.geom_consistency false\
                    --PatchMatchStereo.gpu_index -1\
                    --PatchMatchStereo.num_threads {self._get_num_threads()}',
                    
                f'colmap stereo_fusion\
                    --workspace_path "{os.path.join(colmap_base, "dense")}"\
                    --output_path "{os.path.join(colmap_base, "dense", "fused.ply")}"\
                    --StereoFusion.num_threads {self._get_num_threads()}'
            ])
        
        print(f"Running COLMAP on {'macOS' if self.is_macos else 'CPU'} with {self._get_num_threads()} threads")
        print(f"Skipping dense reconstruction: {skip_dense}")
        
        for i, cmd in enumerate(cmd):
            try:
                print(f"\nCommand {i+1}/{len(cmd)}...")
                
                parts = cmd.split()
                if len(parts) > 1:
                    cmd_name = parts[1]
                    print(f"Running: {cmd_name}")
                
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"Error: {result.stderr[:150]}")
                    
                    if "exhaustive_matcher" in cmd:
                        print("Trying sequential_matcher...")
                        sequential_cmd = cmd.replace("exhaustive_matcher", "sequential_matcher")
                        result = subprocess.run(sequential_cmd, shell=True, capture_output=True, text=True)
                        if result.returncode != 0:
                            return False
                    else:
                        return False
                else:
                    print(f"Command {i+1} completed")
                    
            except Exception as e:
                print(f"Error: {str(e)}")
                return False
        
        self._organize_output(lot_name, size, skip_dense)
        print(f"\nCOLMAP processing completed for {lot_name} at {size}px")
        return True
    
    def convert2text(self, lot_name: str, size: int):
        """_summary_

        Args:
            lot_name (str): _description_
            size (int): _description_
        """
        colmap_base = os.path.join(self.utils.colmap_dir, lot_name, f"{size}px")
        sparse_dir = os.path.join(colmap_base, "sparse", "0")

        if not os.path.exists(sparse_dir):
            print(f"Sparse directory not found: {sparse_dir}")
            return False
        #TODO: change this logic letter for just use the following .bin cameras.bin, images.bin, points3D.bin

        bin_files = [os.path.join(sparse_dir, f) for f in os.listdir(sparse_dir) if f.endswith('.bin')]
        if not any(os.path.exists(f) for f in bin_files):
            print(f"No .bin files found in: {sparse_dir}")
            return False
        try:
            cmd = [
                f'colmap model_converter \
                    --input_path "{sparse_dir}" \
                        --output_path "{sparse_dir}" \
                            --output_type TXT'
            ]
            print(f"Converting COLMAP model to text format...")
            result = subprocess.run(cmd[0], shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error during conversion: {result.stderr[:150]}")
                return False
            return True
        except Exception as e:
            print(f"Exception during conversion: {str(e)}")
            return False
        
    
    
    def _get_num_threads(self):
        if self.is_macos:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            return max(2, cpu_count - 2)  
        else:
            import multiprocessing
            return multiprocessing.cpu_count()
    

    def _organize_output(self, lot_name: str, size: int, skip_dense: bool = True):
        colmap_base = os.path.join(self.utils.colmap_dir, lot_name, f"{size}px")
        output_dir = os.path.join(colmap_base, "output")
        
        os.makedirs(output_dir, exist_ok=True)

        files_to_check = [
            os.path.join(colmap_base, "sparse", "0", "cameras.bin"),
            os.path.join(colmap_base, "sparse", "0", "images.bin"),
            os.path.join(colmap_base, "sparse", "0", "points3D.bin"),
            os.path.join(colmap_base, "sparse", "0", "cameras.txt"),
            os.path.join(colmap_base, "sparse", "0", "images.txt"),
            os.path.join(colmap_base, "sparse", "0", "points3D.txt"),
        ]
        
        copied = False
        for file_path in files_to_check:
            if os.path.exists(file_path):
                shutil.copy2(file_path, output_dir)
                print(f"{os.path.basename(file_path)}")
                copied = True
        
        if not copied:
            print("No output files found")
        else:
            print(f"\nOutput saved to: {output_dir}")