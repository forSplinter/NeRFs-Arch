"""
Evaluation Runner for Archaeological Reconstruction with Cropping
=================================================================
This script provides a command-line interface for evaluating
NeRF reconstructions of archaeological artifacts against ground truth.
Now includes cropping functionality to focus on specific regions.

Usage:
    python evaluate.py --x1 100 --x2 500 --y1 50 --y2 400

The script:
1. Loads ground truth and reconstructed images
2. Crops images to specified region (if coordinates provided)
3. Calculates comprehensive evaluation metrics
4. Generates visual reports
5. Saves results to the 'results' directory
"""

import os
import sys
import argparse

# Add the current directory to the Python path for module import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from archaeological_metrics import evaluate_archaeological_reconstruction

def ensure_directory(path):
    """
    Create directory if it doesn't exist.
    
    Parameters:
    -----------
    path : str
        Directory path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📂 Directory created: {path}")

def parse_crop_coords(args):
    """
    Parse and validate crop coordinates.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    Optional[Tuple[int, int, int, int]]
        Crop coordinates in format (x1, x2, y1, y2) or None
    """
    if args.x1 is not None and args.x2 is not None and args.y1 is not None and args.y2 is not None:
        # Validate coordinates
        if args.x1 >= args.x2:
            print(f"⚠️  Warning: x1 ({args.x1}) should be less than x2 ({args.x2})")
            print("   Adjusting x2 = x1 + 100")
            args.x2 = args.x1 + 100
        
        if args.y1 >= args.y2:
            print(f"⚠️  Warning: y1 ({args.y1}) should be less than y2 ({args.y2})")
            print("   Adjusting y2 = y1 + 100")
            args.y2 = args.y1 + 100
        
        crop_coords = (args.x1, args.x2, args.y1, args.y2)
        print(f"✅ Crop coordinates set: x={args.x1}:{args.x2}, y={args.y1}:{args.y2}")
        return crop_coords
    elif any([args.x1, args.x2, args.y1, args.y2]):
        print("⚠️  Warning: Some crop coordinates provided but not all. Ignoring cropping.")
        return None
    else:
        return None

def main():
    """
    Main function to run archaeological reconstruction evaluation.
    
    Workflow:
    1. Parse command line arguments for crop coordinates
    2. Configure paths for data and results
    3. Verify input files exist
    4. Run evaluation with archaeological metrics (with optional cropping)
    5. Save results and generate reports
    6. Display summary of findings
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate archaeological artifact reconstruction with optional cropping.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py                          # Evaluate without cropping
  python evaluate.py --x1 100 --x2 500 --y1 50 --y2 400  # Evaluate with cropping
  
Coordinate system:
  Origin (0,0) is at top-left of image
  x increases to the right, y increases downward
        """
    )
    
    parser.add_argument('--x1', type=int, help='Left crop coordinate (x1 < x2)')
    parser.add_argument('--x2', type=int, help='Right crop coordinate (x1 < x2)')
    parser.add_argument('--y1', type=int, help='Top crop coordinate (y1 < y2)')
    parser.add_argument('--y2', type=int, help='Bottom crop coordinate (y1 < y2)')
    parser.add_argument('--gt', type=str, default=None, help='Ground truth image path (overrides default)')
    parser.add_argument('--pred', type=str, default=None, help='Predicted image path (overrides default)')
    
    args = parser.parse_args()
    
    # Configure file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "set 2")
    results_dir = os.path.join(base_dir, "results_cropped_2")
    
    # Create results directory if it doesn't exist
    ensure_directory(results_dir)
    
    # Define input file paths (use command line args if provided, else defaults)
    if args.gt:
        gt_path = args.gt
    else:
        gt_path = os.path.join(data_dir, "rock-3.jpg")
        
    if args.pred:
        pred_path = args.pred
    else:
        pred_path = os.path.join(data_dir, "rock-re-3.jpg")
    
    # Parse crop coordinates
    crop_coords = parse_crop_coords(args)
    
    # Define output file paths
    if crop_coords:
        suffix = f"_cropped_x{crop_coords[0]}-{crop_coords[1]}_y{crop_coords[2]}-{crop_coords[3]}"
    else:
        suffix = ""
    
    artifact_base = os.path.splitext(os.path.basename(gt_path))[0]
    report_path = os.path.join(results_dir, f"metric-{artifact_base}{suffix}.png")
    metrics_path = os.path.join(results_dir, f"metric-{artifact_base}{suffix}.txt")
    
    # Print configuration information
    print("=" * 70)
    print("ARCHAEOLOGICAL RECONSTRUCTION EVALUATION SYSTEM")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Ground truth image: {gt_path}")
    print(f"NeRF reconstruction: {pred_path}")
    if crop_coords:
        print(f"Crop region: x={crop_coords[0]}:{crop_coords[1]}, y={crop_coords[2]}:{crop_coords[3]}")
    else:
        print("Crop region: None (using full images)")
    print("=" * 70)
    
    # Verify input files exist
    if not os.path.exists(gt_path):
        print(f"❌ ERROR: File not found: {gt_path}")
        print("Please ensure the ground truth image exists in the specified location.")
        sys.exit(1)
    
    if not os.path.exists(pred_path):
        print(f"❌ ERROR: File not found: {pred_path}")
        print("Please ensure the reconstructed image exists in the specified location.")
        sys.exit(1)
    
    print("✅ All input files found. Starting evaluation...\n")
    
    try:
        # Run the comprehensive evaluation
        results = evaluate_archaeological_reconstruction(
            predicted_image=pred_path,
            ground_truth_image=gt_path,
            crop_coords=crop_coords,
            visualize=True,
            save_path=report_path,
            artifact_name="Dragon Artifact"
        )
        
        # Extract results from the evaluation
        metrics = results['metrics']
        quality = results['quality']
        
        # Display quantitative results in terminal
        print("\n" + "=" * 70)
        print("📊 EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"\n🔢 QUANTITATIVE METRICS:")
        print(f"   PSNR:    {metrics['psnr']:7.2f} dB")
        print(f"   SSIM:    {metrics['ssim']:7.4f}")
        print(f"   MS-SSIM: {metrics['ms_ssim']:7.4f}")
        print(f"   LPIPS:   {metrics['lpips']:7.4f} (lower is better)")
        
        print(f"\n🎯 QUALITY ASSESSMENT:")
        for aspect, assessment in quality.items():
            if aspect != 'quality_assessment':
                # Add emoji indicators based on quality
                emoji = "✅" if "EXCELLENT" in assessment or "GOOD" in assessment else "⚠️"
                print(f"   {emoji} {aspect.upper()}: {assessment}")
        
        # Save detailed metrics to text file
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("EVALUATION RESULTS - Dragon Artifact\n")
            if crop_coords:
                f.write(f"Cropped: x={crop_coords[0]}:{crop_coords[1]}, y={crop_coords[2]}:{crop_coords[3]}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("QUANTITATIVE METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"PSNR:     {metrics['psnr']:.2f} dB\n")
            f.write(f"SSIM:     {metrics['ssim']:.4f}\n")
            f.write(f"MS-SSIM:  {metrics['ms_ssim']:.4f}\n")
            f.write(f"LPIPS:    {metrics['lpips']:.4f}\n\n")
            
            f.write("QUALITY ASSESSMENT:\n")
            f.write("-" * 30 + "\n")
            for aspect, assessment in quality.items():
                if aspect != 'quality_assessment':
                    f.write(f"{aspect.upper()}: {assessment}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("ARCHAEOLOGICAL INTERPRETATION:\n")
            f.write("=" * 50 + "\n")
            
            # Provide archaeological interpretation based on PSNR
            psnr = metrics['psnr']
            if psnr > 40:
                f.write("Exceptional quality - Suitable for archival and publication\n")
            elif psnr > 35:
                f.write("Good quality - Suitable for research and analysis\n")
            elif psnr > 30:
                f.write("Acceptable quality - Basic documentation\n")
            else:
                f.write("Insufficient quality - Requires improvement\n")
            
            # Add crop information if used
            if crop_coords:
                f.write("\n" + "=" * 50 + "\n")
                f.write("CROP INFORMATION\n")
                f.write("=" * 50 + "\n")
                f.write(f"x1 (left):    {crop_coords[0]}\n")
                f.write(f"x2 (right):   {crop_coords[1]}\n")
                f.write(f"y1 (top):     {crop_coords[2]}\n")
                f.write(f"y2 (bottom):  {crop_coords[3]}\n")
                f.write(f"Width:        {crop_coords[1] - crop_coords[0]}\n")
                f.write(f"Height:       {crop_coords[3] - crop_coords[2]}\n")
        
        # Report file saving
        print(f"\n📁 RESULTS SAVED:")
        print(f"   📊 Visual report: {report_path}")
        print(f"   📝 Text metrics: {metrics_path}")
        
        print("\n" + "=" * 70)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        # Display final summary
        artifact_name = "Dragon Artifact" + (" (Cropped)" if crop_coords else "")
        print(f"\n🎯 FINAL SUMMARY FOR '{artifact_name}':")
        print(f"   PSNR Score: {metrics['psnr']:.1f} dB → {quality.get('psnr', 'N/A').split('(')[0]}")
        print(f"   Overall assessment: {quality.get('overall', 'N/A')}")
        
        # Provide actionable recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if metrics['psnr'] < 30:
            print("   • Consider retraining NeRF with more views")
            print("   • Check camera pose estimation accuracy")
            print("   • Verify image capture conditions")
            if not crop_coords:
                print("   • Try cropping to remove background (use --x1 --x2 --y1 --y2)")
        elif metrics['psnr'] < 35:
            print("   • Reconstruction is suitable for research")
            print("   • Consider fine-tuning NeRF parameters")
            print("   • Validate with additional test views")
        else:
            print("   • Excellent reconstruction quality")
            print("   • Suitable for publication and archival")
        
    except Exception as e:
        # Handle any errors during evaluation
        print(f"\n❌ ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    """
    Entry point for the evaluation script.
    """
    main()