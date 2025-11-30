import os
import sys
import subprocess

def main():
    lot_name = os.getenv('LOT_NAME', 'lot1')
    size = int(os.getenv('SIZE', '1365'))
    
    print(f"Starting COLMAP cloud pipeline for {lot_name} {size}px")
    
    from utils.colmap_utils import ColmapUtils
    from utils.colmap_process import ColmapProcess
    
    utils = ColmapUtils('/workspace/dataset')
    processor = ColmapProcess('/workspace/dataset')
    
    colmap_base = utils.setup_colmap(lot_name, size)
    
    input_images = f"/workspace/dataset/colmap/{lot_name}/{size}px/images"
    if not os.path.exists(input_images):
        print(f"Error: Input images not found at {input_images}")
        return False
    
    commands = [
        f'colmap feature_extractor \
            --database_path "{colmap_base}/database.db" \
            --image_path "{input_images}" \
            --SiftExtraction.use_gpu 1 \
            --SiftExtraction.gpu_index 0',
            
        f'colmap exhaustive_matcher \
            --database_path "{colmap_base}/database.db" \
            --SiftMatching.use_gpu 1 \
            --SiftMatching.gpu_index 0',
            
        f'colmap mapper \
            --database_path "{colmap_base}/database.db" \
            --image_path "{input_images}" \
            --output_path "{colmap_base}/sparse" \
            --Mapper.ba_global_use_gpu 1',
            
        f'colmap image_undistorter \
            --image_path "{input_images}" \
            --input_path "{colmap_base}/sparse/0" \
            --output_path "{colmap_base}/dense"',
            
        f'colmap patch_match_stereo \
            --workspace_path "{colmap_base}/dense" \
            --PatchMatchStereo.geom_consistency true \
            --PatchMatchStereo.gpu_index 0',
            
        f'colmap stereo_fusion \
            --workspace_path "{colmap_base}/dense" \
            --output_path "{colmap_base}/dense/fused.ply"'
    ]
    
    for i, cmd in enumerate(commands, 1):
        print(f"Running COLMAP step {i}/6")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Command failed: {result.stderr}")
            return False
    
    print(f"COLMAP cloud pipeline completed for {lot_name} at {size}px")
    return True

if __name__ == "__main__":
    main()