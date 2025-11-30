import argparse
from utils.colmap_utils import ColmapUtils
from utils.colmap_process import ColmapProcess

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
    
    def prepare_gui(self, lot_name: str, size: int):
        if not self.utils.copy_mask(lot_name, size):
            return False
        project_file = self.utils.create_colmap_project(lot_name, size)
        print(f"COLMAP GUI ready. Project file: {project_file}")
        return True
    
    def run_local_reconstruction(self, lot_name: str, size: int):
        success = self.processor.run_colmap_local(lot_name, size)
        if success:
            print(f"Reconstruction completed for Lot: {lot_name}, Size: {size}px")
        else:
            print(f"Reconstruction failed for Lot: {lot_name}, Size: {size}px")
    
    def prepare_cloud_data(self, lot_name: str, size: int):
        colmap_base = self.processor.prepare_cloud_data(lot_name, size)
        if colmap_base:
            print(f"Cloud data prepared at: {colmap_base}")
            return colmap_base
        return None

def main():
    parser = argparse.ArgumentParser(description="COLMAP Manager")
    parser.add_argument("--list", action="store_true", help="List all projects and sizes")
    parser.add_argument("--prepare", help="Prepare COLMAP GUI for a specific project and size")
    parser.add_argument("--local", help="Run local reconstruction for specific project and size")
    parser.add_argument("--cloud-prepare", help="Prepare data for cloud processing")
    
    args = parser.parse_args()
    manager = ColmapManager(dataset_root="dataset")
    
    if args.list:
        manager.list_projects()
    elif args.prepare:
        lot_name, size = args.prepare.split(",")
        manager.prepare_gui(lot_name.strip(), int(size.strip()))
    elif args.local:
        lot_name, size = args.local.split(",")
        manager.run_local_reconstruction(lot_name.strip(), int(size.strip()))
    elif args.cloud_prepare:
        lot_name, size = args.cloud_prepare.split(",")
        manager.prepare_cloud_data(lot_name.strip(), int(size.strip()))
    else:
        parser.print_help()
        
if __name__ == "__main__":
    main()