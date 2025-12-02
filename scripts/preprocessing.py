import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from utils.rmbg import BackgroundRemoval
from utils.imgprocessed import ImageProcessor
from utils.files_utils import copy_files

class Preprocessor:
    def __init__(self, dataset_root: str, max_workers=None):
        """_summary_

        Args:
            dataset_root (str): _description_
            max_workers (_type_, optional): _description_. Defaults to None.
        """
        self.dataset_root = dataset_root
        self.raw_dir = os.path.join(dataset_root, "raw")
        self.processed_dir = os.path.join(dataset_root, "processed")
        self.max_workers = max_workers if max_workers is not None else os.cpu_count()
        os.makedirs(self.processed_dir, exist_ok=True)
        self.setup_directories()

    def setup_directories(self):
        """_summary_

        Args:
            None

        Returns:
            None
        """
        base_folders = ["images", "masks", "resized", "test"]
        test_folders = ["originals", "masked", "comparison"]
        for folder in base_folders:
            full_path = os.path.join(self.processed_dir, folder)
            os.makedirs(full_path, exist_ok=True)
        for folder in test_folders:
            full_path = os.path.join(self.processed_dir, "test", folder)
            os.makedirs(full_path, exist_ok=True)

    def get_lot_folders(self):
        """_summary_

        Args:
            None

        Returns:
            List[str]: _description_
        """
        lot_folders = []
        if os.path.exists(self.raw_dir):
            for item in os.listdir(self.raw_dir):
                item_path = os.path.join(self.raw_dir, item)
                if os.path.isdir(item_path) and item.startswith("lot"):
                    lot_folders.append(item)
        return sorted(lot_folders)

    def preprocess_lot(self, lot_name: str, input_dir: str, output_dir: str, max_size: int = 1024):
        """_summary_

        Args:
            lot_name (str): _description_
            input_dir (str): _description_
            output_dir (str): _description_
            max_size (int, optional): _description_. Defaults to 1024.

        Returns:
            None
        """
        os.makedirs(output_dir, exist_ok=True)
        copied_files = copy_files(src_dir=input_dir, dest_dir=output_dir)
        
        def process_image(filename: str):
            """_summary_

            Args:
                filename (str): _description_

            Returns:
                _type_: _description_
            """
            input_path = os.path.join(output_dir, filename)
            ImageProcessor.resize_image(image_path=input_path, output_path=input_path, max_size=max_size)
            return filename
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(
                executor.map(process_image, [os.path.basename(f) for f in copied_files]), 
                total=len(copied_files),
                desc=f"{lot_name} resize {max_size}px"
            ))

    def remove_backgrounds(self, input_dir: str, output_dir: str):
        """_summary_

        Args:
            input_dir (str): _description_
            output_dir (str): _description_
        """
        pipeline = BackgroundRemoval(input_dir=input_dir, output_dir=output_dir, max_workers=self.max_workers)
        pipeline.process_batch()

    def run_pipeline(self, target_sizes=[1365]):
        """_summary_

        Args:
            target_sizes (list, optional): _description_. Defaults to [1365].

        Returns:
            None
        """
        lot_folders = self.get_lot_folders()
        if not lot_folders:
            print(f"No lot folders in {self.raw_dir}")
            return
        
        for lot_name in lot_folders:
            lot_input_dir = os.path.join(self.raw_dir, lot_name)
            images_dir = os.path.join(self.processed_dir, "images", lot_name)
            copy_files(src_dir=lot_input_dir, dest_dir=images_dir)
            
            for size in target_sizes:
                resized_dir = os.path.join(self.processed_dir, "resized", lot_name, f"{size}px")
                masks_dir = os.path.join(self.processed_dir, "masks", lot_name, f"{size}px")
                os.makedirs(resized_dir, exist_ok=True)
                os.makedirs(masks_dir, exist_ok=True)
                
                self.preprocess_lot(
                    lot_name=lot_name,
                    input_dir=images_dir, 
                    output_dir=resized_dir, 
                    max_size=size
                )
                
                self.remove_backgrounds(
                    input_dir=resized_dir, 
                    output_dir=masks_dir
                )

def main():
    preprocessor = Preprocessor(dataset_root="dataset")
    preprocessor.run_pipeline(target_sizes=[1365])


if __name__ == "__main__":
    main()