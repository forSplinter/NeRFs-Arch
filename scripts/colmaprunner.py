import argparse
import sys
from utils.colmap.colmap_utils import ColmapUtils
from utils.colmap.colmap_pipeline import ColmapProcess
from utils.colmap.colmap2nerf import ColmapToNeRFConverter

def main():
    parser = argparse.ArgumentParser(description="COLMAP2NeRF Manager")
    
    parser.add_argument("--list", action="store_true", help="List available projects")
    parser.add_argument("--prepare", help="Prepare project for COLMAP GUI: lot1,1365")
    parser.add_argument("--run", help="Run COLMAP reconstruction: lot1,1365")
    parser.add_argument("--with-dense", action="store_true", help="Include dense reconstruction")
    parser.add_argument("--text", help="Convert .bin to .txt: lot1,1365")
    
    parser.add_argument("--colmap_path", help="Path to COLMAP .bin files")
    parser.add_argument("--text_path", help="Path to COLMAP .txt files")
    parser.add_argument("--images_path", help="Path to images")
    parser.add_argument("--out", default="transforms.json", help="Output file")
    parser.add_argument("--aabb_scale", type=int, default=16, help="AABB scale factor")
    parser.add_argument("--dataset_root", default="dataset", help="Dataset root directory")
    args = parser.parse_args()
    
    utils = ColmapUtils(dataset_root=args.dataset_root)
    processor = ColmapProcess(dataset_root=args.dataset_root)
    converter = ColmapToNeRFConverter()
    
    if args.list:
        lots = utils.get_lots()
        for lot in lots:
            sizes = utils.get_sizes(lot)
            print(f"Lot: {lot}, Sizes: {sizes}")
    
    elif args.prepare:
        try:
            lot_name, size = args.prepare.split(",")
            if utils.copy_mask(lot_name.strip(), int(size.strip())):
                project_file = utils.create_colmap_project(lot_name.strip(), int(size.strip()))
                print(f"Project ready: {project_file}")
            else:
                print("Preparation failed")
        except:
            print("Error: use --prepare lot1,1365")
    
    elif args.run:
        try:
            lot_name, size = args.run.split(",")
            skip_dense = not args.with_dense
            success = processor.run_colmap_local(lot_name.strip(), int(size.strip()), skip_dense)
            print(f"{'Success' if success else 'Failed'}: {lot_name.strip()}, {size.strip()}px")
        except:
            print("Error: use --run lot1,1365")
    
    elif args.text:
        try:
            lot_name, size = args.text.split(",")
            success = processor.convert2text(lot_name.strip(), int(size.strip()))
            print(f"Conversion {'success' if success else 'failed'}: {lot_name.strip()}, {size.strip()}px")
        except:
            print("Error: use --text lot1,1365")
    
    elif args.text_path or args.colmap_path:
        if not args.images_path:
            print("Error: --images_path is required")
            sys.exit(1)
        
        success = converter.convert(
            colmap_binary_path=args.colmap_path,
            colmap_text_path=args.text_path,
            images_directory=args.images_path,
            output_file=args.out,
            aabb_scale=args.aabb_scale
        )
        print(f"NeRF conversion: {'success' if success else 'failed'}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()