import argparse
from utils.colmap.colmap_utils import ColmapUtils
from utils.colmap.colmap_pipeline import ColmapProcess

class ColmapManager:
    def __init__(self, dataset_root: str):
        self.dataset_root = dataset_root
        self.utils = ColmapUtils(dataset_root=dataset_root)
        self.processor = ColmapProcess(dataset_root=dataset_root)

    def list_projects(self):
        lots = self.utils.get_lots()
        for lot in lots:
            sizes = self.utils.get_sizes(lot)
            print(f"Lot: {lot}, Sizes: {sizes}")

    def prepare(self, lot_name: str, size: int):
        if not self.utils.copy_mask(lot_name, size):
            return False
        project_file = self.utils.create_colmap_project(lot_name, size)
        print(f"COLMAP GUI ready. Project file: {project_file}")
        return True

    def run_local(self, lot_name: str, size: int, skip_dense: bool = True):
        print(f"Starting reconstruction for {lot_name} at {size}px...")
        success = self.processor.run_colmap_local(lot_name, size, skip_dense)

        if success:
            print(f"Done: {lot_name}, {size}px")
        else:
            print(f"Failed: {lot_name}, {size}px")

        return success
    
    def convert2text(self, lot_name: str, size: int):
        success = self.processor.convert2text(lot_name, size)
        if success:
            print(f"Conversion to text completed for {lot_name} at {size}px")
        else:
            print(f"Conversion to text failed for {lot_name} at {size}px")
        
        return success 


def main():
    parser = argparse.ArgumentParser(description="COLMAP Manager")
    parser.add_argument("--list", action="store_true", help="List projects")
    parser.add_argument("--prepare", help="Prepare for COLMAP GUI: lot1,1365")
    parser.add_argument("--run", help="Run reconstruction: lot1,1365")
    parser.add_argument(
        "--with-dense", action="store_true", help="Run dense reconstruction (slower)"
    )
    parser.add_argument("--text", help="Convert model to text format")
    args = parser.parse_args()
    manager = ColmapManager(dataset_root="dataset")

    if args.list:
        manager.list_projects()
    elif args.prepare:
        try:
            lot_name, size = args.prepare.split(",")
            manager.prepare(lot_name.strip(), int(size.strip()))
        except:
            print("Error: use --prepare lot1,1365")
    elif args.text:
        try:
            lot_name, size = args.text.split(",")
            manager.convert2text(lot_name.strip(), int(size.strip()))
        except:
            print("Error: use --text lot1,1365")
    elif args.run:
        try:
            lot_name, size = args.run.split(",")
            skip_dense = not args.with_dense
            manager.run_local(lot_name.strip(), int(size.strip()), skip_dense)
        except:
            print("Error: use --run lot1,1365")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
