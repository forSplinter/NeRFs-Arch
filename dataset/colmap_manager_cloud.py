import os
import sys
import subprocess

def main():
    lot_name = os.getenv('LOT_NAME', 'lot1')
    size = int(os.getenv('SIZE', '1365'))
    
    print(f"[COLMAP] Starting for lot: {lot_name}, size: {size}")
    
    dataset_root = '/workspace/dataset'
    colmap_base = f"{dataset_root}/colmap/{lot_name}/{size}px"
    database_path = f"{colmap_base}/database.db"
    input_images = f"{colmap_base}/images"
    
    print(f"[COLMAP] Base directory: {colmap_base}")
    print(f"[COLMAP] Images directory: {input_images}")
    
    if not os.path.exists(input_images):
        print(f"[ERROR] Images directory does not exist: {input_images}")
        return False
    
    images_count = len([f for f in os.listdir(input_images) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    print(f"[COLMAP] Found {images_count} images")
    
    if images_count == 0:
        print(f"[ERROR] No images found in {input_images}")
        return False
    
    # Créer les dossiers nécessaires
    os.makedirs(f"{colmap_base}/dense", exist_ok=True)
    os.makedirs(f"{colmap_base}/sparse", exist_ok=True)
    
    commands = [
        f'colmap feature_extractor \
            --database_path "{database_path}" \
            --image_path "{input_images}" \
            --ImageReader.single_camera 1 \
            --SiftExtraction.use_gpu 1 \
            --SiftExtraction.gpu_index 0',
            
        f'colmap exhaustive_matcher \
            --database_path "{database_path}" \
            --SiftMatching.use_gpu 1 \
            --SiftMatching.gpu_index 0',
            
        f'colmap mapper \
            --database_path "{database_path}" \
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
    
    step_names = [
        "feature_extractor",
        "exhaustive_matcher",
        "mapper",
        "image_undistorter",
        "patch_match_stereo",
        "stereo_fusion"
    ]
    
    for i, (step, cmd) in enumerate(zip(step_names, commands)):
        print(f"\n[COLMAP] Step {i+1}/6: {step}")
        print(f"[COLMAP] Command: {cmd.split()[0]}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[ERROR] Step {step} failed")
            print(f"[ERROR] Return code: {result.returncode}")
            print(f"[ERROR] Stderr: {result.stderr[:500]}")  # Limité à 500 caractères
            return False
        
        print(f"[COLMAP] {step} completed")
    
    # Vérifier si le résultat final existe
    final_ply = f"{colmap_base}/dense/fused.ply"
    if os.path.exists(final_ply):
        print(f"\n[COLMAP] Success! Point cloud created: {final_ply}")
        print(f"[COLMAP] File size: {os.path.getsize(final_ply)} bytes")
    else:
        print(f"\n[WARNING] Final PLY file not found: {final_ply}")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[COLMAP] ✓ Pipeline completed successfully")
    else:
        print("\n[COLMAP] ✗ Pipeline failed")
    sys.exit(0 if success else 1)