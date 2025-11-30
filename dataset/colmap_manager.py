import argparse
from utils.colmap_utils import ColmapUtils
from utils.colmap_process import ColmapProcess

class ColmapManager:
    def __init__(self, dataset_root: str):
        """_summary_

        Args:
            dataset_root (str): _description_
        """
        self.dataset_root = dataset_root
        self.utils = ColmapUtils(dataset_root=dataset_root)
        self.processor = ColmapProcess(dataset_root=dataset_root)
    
    def list_projects(self):
        """_summary_
        """
        lots = self.utils.get_lots()
        for lot in lots:
            sizes = self.utils.get_sizes(lot)
            print(f"Lot: {lot}, Sizes: {sizes}")
    
    def prepare_gui(self, lot_name: str, size: int):
        """_summary_

        Args:
            lot_name (str): _description_
            size (int): _description_
        """
        if not self.utils.copy_mask(lot_name, size):
            return False
        colmap_base = self.utils.setup_colmap(lot_name, size)
        print(f"COLMAP GUI ready at: {colmap_base}")
        return True
    
    def run_reconstruction(self, lot_name: str, size: int):
        """_summary_

        Args:
            lot_name (str): _description_
            size (int): _description_
        """
        success = self.processor.run_colmap(lot_name, size)
        if success:
            print(f"Reconstruction completed for Lot: {lot_name}, Size: {size}px")
        else:
            print(f"Reconstruction failed for Lot: {lot_name}, Size: {size}px")

def main():
    parser = argparse.ArgumentParser(description="COLMAP Manager")
    parser.add_argument("--list", action="store_true", help="List all projects and sizes")
    parser.add_argument("--prepare", help="Prepare COLMAP GUI for a specific project and size")
    parser.add_argument("--auto", help="Automatically run reconstruction for all projects and sizes")
    args = parser.parse_args()
    manager = ColmapManager(dataset_root="dataset")
    if args.list:
        manager.list_projects()
    elif args.prepare:
        lot_name, size = args.prepare.split(",")
        manager.prepare_gui(lot_name.strip(), int(size.strip()))
    elif args.auto:
        lot_name, size = args.auto.split(",")
        manager.run_reconstruction(lot_name.strip(), int(size.strip()))
    else:
        parser.print_help()
        
if __name__ == "__main__":
    main()