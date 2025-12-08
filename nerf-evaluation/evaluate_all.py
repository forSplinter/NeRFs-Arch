"""
Multi-Set Archaeological Reconstruction Evaluator
=================================================
This script automatically evaluates all NeRF reconstruction pairs
across multiple dataset folders (set 1, set 2, etc.).

Features:
- Automatic discovery of dataset folders
- Case-insensitive file extension handling (.jpg, .JPG, .png, .PNG)
- Dynamic pairing of ground truth and reconstructed images
- Separate results folders for each dataset
- CSV summary for cross-dataset comparison
- Progress tracking and error handling
"""

import os
import sys
import glob
import re
import csv
import json
from typing import List, Dict, Tuple
from datetime import datetime

# Add current directory to Python path for module import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from archaeological_metrics import evaluate_archaeological_reconstruction

def ensure_directory(path: str) -> str:
    """
    Create directory if it doesn't exist and return the path.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"📂 Directory created: {path}")
    return path

def extract_set_number(set_path: str) -> int:
    """
    Extract set number from folder name.
    """
    folder_name = os.path.basename(set_path)
    numbers = re.findall(r'\d+', folder_name)
    return int(numbers[0]) if numbers else 0

def find_artifact_pairs_case_insensitive(set_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find all ground truth and reconstruction pairs in a set directory.
    This version is case-insensitive for file extensions.
    
    Supports: .jpg, .JPG, .jpeg, .JPEG, .png, .PNG
    """
    artifact_pairs = []
    
    # First, get all files in the directory with case-insensitive filtering
    all_files = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    for root, dirs, files in os.walk(set_dir):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in ['.jpg', '.jpeg', '.png']:
                all_files.append(file)
    
    if not all_files:
        print(f"   No image files found in {set_dir}")
        print(f"   Supported formats: .jpg, .jpeg, .png (case-insensitive)")
        return artifact_pairs
    
    # Group files by base name and number
    file_dict = {}
    
    for filename in all_files:
        # Try multiple patterns to match filenames
        patterns = [
            r'^(.+)-(\d+)\.(jpg|jpeg|png|JPG|JPEG|PNG)$',  # artifact-1.jpg
            r'^(.+)_(\d+)\.(jpg|jpeg|png|JPG|JPEG|PNG)$',  # artifact_1.jpg
            r'^(.+)\.(\d+)\.(jpg|jpeg|png|JPG|JPEG|PNG)$'  # artifact.1.jpg
        ]
        
        matched = False
        for pattern in patterns:
            match = re.match(pattern, filename, re.IGNORECASE)
            if match:
                artifact_base = match.group(1)  # e.g., "rock", "dragon"
                artifact_num = match.group(2)   # e.g., "1", "2", "3"
                extension = match.group(3)      # e.g., "jpg", "JPG", "png"
                
                key = (artifact_base.lower(), artifact_num)
                
                if key not in file_dict:
                    file_dict[key] = {}
                
                # Check if this is ground truth or reconstruction
                if '-re-' in filename.lower() or '_re_' in filename.lower():
                    file_dict[key]['reconstruction'] = filename
                else:
                    file_dict[key]['ground_truth'] = filename
                
                matched = True
                break
        
        if not matched:
            print(f"   ⚠️  Could not parse filename: {filename}")
    
    # Now create pairs from the dictionary
    for (artifact_base, artifact_num), files in file_dict.items():
        if 'ground_truth' in files and 'reconstruction' in files:
            gt_file = files['ground_truth']
            re_file = files['reconstruction']
            
            # Construct full paths
            gt_path = os.path.join(set_dir, gt_file)
            re_path = os.path.join(set_dir, re_file)
            
            # Create artifact name (preserving original case from ground truth)
            gt_base_name = os.path.splitext(gt_file)[0]
            artifact_name = re.sub(r'-\d+$', '', gt_base_name)  # Remove trailing number
            artifact_name = f"{artifact_name}-{artifact_num}"
            
            artifact_pairs.append((gt_path, re_path, artifact_name))
        else:
            if 'ground_truth' in files:
                print(f"   ⚠️  Missing reconstruction for: {files['ground_truth']}")
            elif 'reconstruction' in files:
                print(f"   ⚠️  Missing ground truth for: {files['reconstruction']}")
    
    return artifact_pairs

def find_artifact_pairs_simple(set_dir: str) -> List[Tuple[str, str, str]]:
    """
    Simple version that finds pairs based on common naming patterns.
    Case-insensitive and handles various naming conventions.
    """
    artifact_pairs = []
    
    # Get all image files in the directory
    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(set_dir, ext)))
    
    # Create a mapping of base names to file paths
    file_map = {}
    
    for image_path in all_images:
        filename = os.path.basename(image_path)
        
        # Extract base name and number (case-insensitive)
        # Try to match patterns like: rock-1.jpg, rock-re-1.jpg, ROCK-1.JPG
        filename_lower = filename.lower()
        
        # Check if it's a reconstruction file
        is_reconstruction = '-re-' in filename_lower or '_re_' in filename_lower
        
        # Remove reconstruction marker for matching
        match_filename = filename_lower.replace('-re-', '-').replace('_re_', '_')
        
        # Extract base name and number
        # Pattern: something-NUMBER.extension
        pattern = r'^(.+?)[-_]?(\d+)\.(jpg|jpeg|png)$'
        match = re.match(pattern, match_filename)
        
        if match:
            base_name = match.group(1)  # e.g., "rock", "dragon"
            number = match.group(2)     # e.g., "1", "2"
            ext = match.group(3)        # e.g., "jpg"
            
            key = f"{base_name}-{number}"
            
            if key not in file_map:
                file_map[key] = {'gt': None, 're': None}
            
            if is_reconstruction:
                file_map[key]['re'] = image_path
            else:
                file_map[key]['gt'] = image_path
    
    # Create pairs from the mapping
    for key, files in file_map.items():
        if files['gt'] and files['re']:
            # Extract nice artifact name
            base_name, number = key.split('-')
            artifact_name = f"{base_name}-{number}"
            artifact_pairs.append((files['gt'], files['re'], artifact_name))
        else:
            if files['gt'] and not files['re']:
                print(f"   ⚠️  Missing reconstruction for: {os.path.basename(files['gt'])}")
            elif files['re'] and not files['gt']:
                print(f"   ⚠️  Missing ground truth for: {os.path.basename(files['re'])}")
    
    return artifact_pairs

def find_artifact_pairs_direct(set_dir: str) -> List[Tuple[str, str, str]]:
    """
    Direct approach: Look for ground truth files and find matching reconstructions.
    Handles case-insensitive extensions and various naming patterns.
    """
    artifact_pairs = []
    
    # Define all possible ground truth patterns
    gt_patterns = [
        # Format: {name}-{number}.{ext} (case-insensitive)
        r'^(.+)-(\d+)\.(jpg|jpeg|png|JPG|JPEG|PNG)$',
        r'^(.+)_(\d+)\.(jpg|jpeg|png|JPG|JPEG|PNG)$',
        r'^(.+)\.(\d+)\.(jpg|jpeg|png|JPG|JPEG|PNG)$',
    ]
    
    # Get all files in the directory
    all_files = os.listdir(set_dir)
    
    # First pass: Identify ground truth files
    gt_files = []
    for filename in all_files:
        # Skip reconstruction files in this pass
        if '-re-' in filename.lower() or '_re_' in filename.lower():
            continue
            
        for pattern in gt_patterns:
            match = re.match(pattern, filename, re.IGNORECASE)
            if match:
                artifact_base = match.group(1)  # e.g., "rock", "dragon"
                artifact_num = match.group(2)   # e.g., "1", "2", "3"
                extension = match.group(3)      # e.g., "jpg", "JPG"
                
                gt_files.append({
                    'filename': filename,
                    'base': artifact_base,
                    'num': artifact_num,
                    'ext': extension
                })
                break
    
    # Second pass: Find matching reconstructions
    for gt_info in gt_files:
        # Try multiple reconstruction filename patterns
        re_patterns = [
            f"{gt_info['base']}-re-{gt_info['num']}.{gt_info['ext']}",  # rock-re-1.JPG
            f"{gt_info['base']}-re-{gt_info['num']}.{gt_info['ext'].lower()}",  # rock-re-1.jpg
            f"{gt_info['base']}-re-{gt_info['num']}.{gt_info['ext'].upper()}",  # rock-re-1.JPG
            f"{gt_info['base']}_re_{gt_info['num']}.{gt_info['ext']}",  # rock_re_1.JPG
            f"{gt_info['base']}_re_{gt_info['num']}.{gt_info['ext'].lower()}",  # rock_re_1.jpg
            f"{gt_info['base']}_re_{gt_info['num']}.{gt_info['ext'].upper()}",  # rock_re_1.JPG
        ]
        
        found = False
        for re_pattern in re_patterns:
            re_path = os.path.join(set_dir, re_pattern)
            if os.path.exists(re_path):
                gt_path = os.path.join(set_dir, gt_info['filename'])
                artifact_name = f"{gt_info['base']}-{gt_info['num']}"
                artifact_pairs.append((gt_path, re_path, artifact_name))
                found = True
                break
        
        if not found:
            print(f"   ⚠️  Could not find reconstruction for: {gt_info['filename']}")
            print(f"      Tried patterns: {gt_info['base']}-re-{gt_info['num']}.[ext]")
    
    return artifact_pairs

def evaluate_dataset(set_dir: str, set_number: int, base_results_dir: str) -> Dict[str, Dict]:
    """
    Evaluate all artifact pairs in a single dataset.
    """
    print(f"\n{'='*70}")
    print(f"📊 EVALUATING SET {set_number}: {os.path.basename(set_dir)}")
    print(f"{'='*70}")
    
    # Create results directory for this set
    set_results_dir = os.path.join(base_results_dir, f"results_{set_number}")
    ensure_directory(set_results_dir)
    
    # Find all artifact pairs in this set (using case-insensitive method)
    artifact_pairs = find_artifact_pairs_direct(set_dir)
    
    if not artifact_pairs:
        print(f"❌ No artifact pairs found in {set_dir}")
        print("\n💡 EXPECTED NAMING CONVENTIONS:")
        print("   Ground truth:    rock-1.jpg, dragon-2.png, artifact-3.JPG")
        print("   Reconstruction:  rock-re-1.jpg, dragon-re-2.png, artifact-re-3.JPG")
        print("\n   Also supported:")
        print("   - rock_1.jpg / rock_re_1.jpg")
        print("   - rock.1.jpg / rock.re.1.jpg")
        print("   - Case-insensitive: .JPG, .jpg, .JPEG, .jpeg, .PNG, .png")
        
        # List what files we found
        print(f"\n📁 Files found in {set_dir}:")
        all_files = os.listdir(set_dir)
        if all_files:
            for file in sorted(all_files):
                file_lower = file.lower()
                if file_lower.endswith(('.jpg', '.jpeg', '.png')):
                    print(f"   • {file}")
        else:
            print("   (No files found)")
        
        return {}
    
    print(f"✅ Found {len(artifact_pairs)} artifact pairs to evaluate:")
    for _, _, artifact_name in artifact_pairs:
        print(f"   • {artifact_name}")
    
    results = {}
    
    # Evaluate each artifact pair
    for gt_path, re_path, artifact_name in artifact_pairs:
        print(f"\n{'─'*60}")
        print(f"🔍 Evaluating: {artifact_name}")
        print(f"   Ground truth: {os.path.basename(gt_path)}")
        print(f"   Reconstruction: {os.path.basename(re_path)}")
        
        try:
            # Define output paths
            report_filename = f"metric-{artifact_name}.png"
            text_filename = f"metric-{artifact_name}.txt"
            json_filename = f"metric-{artifact_name}.json"
            
            report_path = os.path.join(set_results_dir, report_filename)
            text_path = os.path.join(set_results_dir, text_filename)
            json_path = os.path.join(set_results_dir, json_filename)
            
            # Run evaluation
            eval_results = evaluate_archaeological_reconstruction(
                predicted_image=re_path,
                ground_truth_image=gt_path,
                visualize=True,
                save_path=report_path,
                artifact_name=f"{artifact_name} (Set {set_number})"
            )
            
            # Save detailed text report
            metrics = eval_results['metrics']
            quality = eval_results['quality']
            
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write(f"ARCHAEOLOGICAL EVALUATION REPORT\n")
                f.write(f"Artifact: {artifact_name} (Set {set_number})\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("FILE PATHS:\n")
                f.write(f"  Ground truth: {gt_path}\n")
                f.write(f"  Reconstruction: {re_path}\n\n")
                
                f.write("QUANTITATIVE METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"PSNR:     {metrics['psnr']:.2f} dB\n")
                f.write(f"SSIM:     {metrics['ssim']:.4f}\n")
                f.write(f"MS-SSIM:  {metrics['ms_ssim']:.4f}\n")
                f.write(f"LPIPS:    {metrics['lpips']:.4f}\n\n")
                
                f.write("QUALITY ASSESSMENT:\n")
                f.write("-" * 40 + "\n")
                for aspect, assessment in quality.items():
                    if aspect != 'quality_assessment':
                        f.write(f"{aspect.upper()}: {assessment}\n")
                
                f.write("\nARCHAEOLOGICAL INTERPRETATION:\n")
                f.write("-" * 40 + "\n")
                if metrics['psnr'] > 40:
                    f.write("Exceptional quality - Archival grade\n")
                elif metrics['psnr'] > 35:
                    f.write("Good quality - Research grade\n")
                elif metrics['psnr'] > 30:
                    f.write("Acceptable quality - Documentation grade\n")
                else:
                    f.write("Insufficient quality - Needs improvement\n")
            
            # Save JSON data for programmatic access
            json_data = {
                "artifact": artifact_name,
                "set": set_number,
                "timestamp": datetime.now().isoformat(),
                "files": {
                    "ground_truth": gt_path,
                    "reconstruction": re_path
                },
                "metrics": metrics,
                "quality": quality,
                "outputs": {
                    "visual_report": report_path,
                    "text_report": text_path,
                    "json_data": json_path
                }
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            
            # Store results for summary
            results[artifact_name] = {
                "psnr": metrics['psnr'],
                "ssim": metrics['ssim'],
                "ms_ssim": metrics['ms_ssim'],
                "lpips": metrics['lpips'],
                "quality": quality['overall'],
                "report": report_path
            }
            
            print(f"✅ Evaluation completed for {artifact_name}")
            print(f"   PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}")
            print(f"   Files saved in: {set_results_dir}")
            
        except Exception as e:
            print(f"❌ Error evaluating {artifact_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def create_summary_csv(all_results: Dict[int, Dict[str, Dict]], output_path: str):
    """
    Create a CSV summary of all evaluation results.
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Set', 'Artifact', 'PSNR (dB)', 'SSIM', 'MS-SSIM', 'LPIPS', 
                     'PSNR Grade', 'SSIM Grade', 'LPIPS Grade', 'Overall Quality', 'Report Path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for set_num, set_results in all_results.items():
            for artifact_name, artifact_results in set_results.items():
                psnr = artifact_results['psnr']
                ssim_val = artifact_results['ssim']
                lpips_val = artifact_results['lpips']
                
                psnr_grade = "✅" if psnr > 35 else "⚠️" if psnr > 30 else "❌"
                ssim_grade = "✅" if ssim_val > 0.85 else "⚠️" if ssim_val > 0.70 else "❌"
                lpips_grade = "✅" if lpips_val < 0.20 else "⚠️" if lpips_val < 0.30 else "❌"
                
                writer.writerow({
                    'Set': set_num,
                    'Artifact': artifact_name,
                    'PSNR (dB)': f"{psnr:.2f}",
                    'SSIM': f"{ssim_val:.4f}",
                    'MS-SSIM': f"{artifact_results['ms_ssim']:.4f}",
                    'LPIPS': f"{lpips_val:.4f}",
                    'PSNR Grade': psnr_grade,
                    'SSIM Grade': ssim_grade,
                    'LPIPS Grade': lpips_grade,
                    'Overall Quality': artifact_results['quality'],
                    'Report Path': artifact_results['report']
                })

def generate_summary_report(all_results: Dict[int, Dict[str, Dict]], output_path: str):
    """
    Generate a comprehensive summary report of all evaluations.
    """
    total_artifacts = sum(len(results) for results in all_results.values())
    total_sets = len(all_results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ARCHAEOLOGICAL RECONSTRUCTION EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Sets Evaluated: {total_sets}\n")
        f.write(f"Total Artifacts Evaluated: {total_artifacts}\n\n")
        
        for set_num, set_results in all_results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"SET {set_num} RESULTS\n")
            f.write(f"{'='*60}\n\n")
            
            set_psnr_values = [r['psnr'] for r in set_results.values()]
            set_ssim_values = [r['ssim'] for r in set_results.values()]
            set_lpips_values = [r['lpips'] for r in set_results.values()]
            
            f.write(f"Number of artifacts: {len(set_results)}\n")
            f.write(f"Average PSNR: {sum(set_psnr_values)/len(set_psnr_values):.2f} dB\n")
            f.write(f"Average SSIM: {sum(set_ssim_values)/len(set_ssim_values):.4f}\n")
            f.write(f"Average LPIPS: {sum(set_lpips_values)/len(set_lpips_values):.4f}\n\n")
            
            f.write("Artifact Details:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Artifact':<15} {'PSNR (dB)':<12} {'SSIM':<10} {'Quality':<20}\n")
            f.write("-" * 60 + "\n")
            
            for artifact_name, artifact_results in set_results.items():
                f.write(f"{artifact_name:<15} {artifact_results['psnr']:<12.2f} ")
                f.write(f"{artifact_results['ssim']:<10.4f} {artifact_results['quality']:<20}\n")

def main():
    """
    Main function to evaluate all datasets.
    """
    print("\n" + "="*70)
    print("🏛️  MULTI-SET ARCHAEOLOGICAL RECONSTRUCTION EVALUATOR")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Base directory: {base_dir}")
    
    main_results_dir = os.path.join(base_dir, "results")
    ensure_directory(main_results_dir)
    
    # Find all dataset folders (case-insensitive)
    set_patterns = ["set*", "Set*", "SET*", "dataset*", "Dataset*", "data*", "Data*"]
    set_dirs = []
    
    for pattern in set_patterns:
        set_dirs.extend(glob.glob(os.path.join(base_dir, pattern)))
    
    # Filter to only include directories
    set_dirs = [d for d in set_dirs if os.path.isdir(d)]
    
    # Sort sets by extracted number
    set_dirs.sort(key=lambda x: extract_set_number(x))
    
    if not set_dirs:
        print("\n❌ No dataset folders found!")
        print("Expected folders named: 'set 1', 'set 2', 'Set 1', 'SET_2', etc.")
        print("Case-insensitive search for folders starting with 'set', 'dataset', or 'data'")
        print(f"\nCurrent directory contents: {os.listdir(base_dir)}")
        sys.exit(1)
    
    print(f"\n📁 Found {len(set_dirs)} dataset folder(s):")
    for i, set_dir in enumerate(set_dirs, 1):
        set_name = os.path.basename(set_dir)
        set_num = extract_set_number(set_dir)
        print(f"   {i}. {set_name} (Set {set_num})")
    
    # Store all results
    all_results = {}
    
    # Evaluate each dataset
    for set_dir in set_dirs:
        set_num = extract_set_number(set_dir)
        set_name = os.path.basename(set_dir)
        
        set_results = evaluate_dataset(set_dir, set_num, main_results_dir)
        
        if set_results:
            all_results[set_num] = set_results
            print(f"\n✅ Completed evaluation of Set {set_num} ({set_name})")
            print(f"   Results saved in: {main_results_dir}/results_{set_num}/")
    
    # Generate summary reports
    if all_results:
        print(f"\n{'='*70}")
        print("📈 GENERATING SUMMARY REPORTS")
        print(f"{'='*70}")
        
        csv_path = os.path.join(main_results_dir, "evaluation_summary.csv")
        create_summary_csv(all_results, csv_path)
        print(f"✅ CSV summary created: {csv_path}")
        
        report_path = os.path.join(main_results_dir, "evaluation_summary.txt")
        generate_summary_report(all_results, report_path)
        print(f"✅ Detailed report created: {report_path}")
        
        total_artifacts = sum(len(results) for results in all_results.values())
        print(f"\n{'='*70}")
        print("🎉 EVALUATION COMPLETE!")
        print(f"{'='*70}")
        print(f"Total datasets evaluated: {len(all_results)}")
        print(f"Total artifacts evaluated: {total_artifacts}")
        print(f"\nResults are organized in: {main_results_dir}")
    else:
        print("\n❌ No artifacts were successfully evaluated.")

if __name__ == "__main__":
    main()