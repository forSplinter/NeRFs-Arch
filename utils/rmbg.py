from glob import glob
import os 
import sys 
import cv2 
import numpy as np 
from PIL import Image
import rembg
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm

class BackgroundRemoval:
    def __init__(self, input_dir: str, output_dir: str, max_workers=None):
        """_summary_

        Args:
            input_dir (_type_): _description_
            output_dir (_type_): _description_
            max_workers (_type_, optional): _description_. Defaults to None.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_workers = max_workers if max_workers is not None else os.cpu_count()
        os.makedirs(output_dir, exist_ok=True)
        self.rembg_lock = threading.Lock()
    
    def get_images(self, extensions: tuple = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')) -> list:
        """Get list of image files with specific extensions in the input directory.

        Args:
            extension (str): Image file extension (e.g., '.jpg', '.png').

        Returns:
            list: List of image file paths.
        """
        image_files = []
        for ext in extensions:
            pattern = os.path.join(self.input_dir, ext)
            image_files.extend(glob(pattern))
        return sorted(image_files)
    
    def remove_background(self, image_path: str, output_path: str):
        """_summary_

        Args:
            image_path (str): _description_
            output_path (str): _description_
        """
        try:
            with open(image_path, 'rb') as i:
                input_data = i.read()
            with self.rembg_lock:
                output_data = rembg.remove(input_data)
            with open(output_path, 'wb') as o:
                o.write(output_data)
            return True, image_path, None
        except Exception as e:
            return False, image_path, str(e)
    
    def process_images(self, image_path: str):
        """_summary_

        Args:
            image_path (str): _description_
        """
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(self.output_dir, f"{name}.png")

        success, file_path, error = self.remove_background(image_path, output_path)
        if success:
            return True, file_path, None
        else:
            return False, file_path, error
    
    def process_batch(self, use_multiprocessing: bool = False, batch_size: int = None):
        """_summary_

        Args:
            use_multiprocessing (bool, optional): _description_. Defaults to False.
            batch_size (int, optional): _description_. Defaults to None.
        """
        image_files = self.get_images()
        if not image_files:
            print("No images found in the input directory.")
            return
        total_images = len(image_files)
        print(f"Total images to process: {total_images}")
        print(f"{self.max_workers} workers")

        if batch_size and batch_size > 0:
            batches = [image_files[i:i + batch_size] for i in range(0, total_images, batch_size)]
        else:
            batches = [image_files]
        
        all_results = []
        for batch_num, batch in enumerate(batches):
            print(f"Processing batch {batch_num + 1}/{len(batches)} with {len(batch)} images...")
            if use_multiprocessing:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    results = list(tqdm(executor.map(self.process_images, batch), total=len(batch)))
            else:
                results = []
                for image_path in tqdm(batch, total=len(batch)):
                    result = self.process_images(image_path)
                    results.append(result)
            all_results.extend(results)
        
        self._print_summary(all_results)
    
    def _print_summary(self, results: list):
        """_summary_

        Args:
            results (list): _description_
        """
        success_count = sum(1 for r in results if r[0])
        failure_count = len(results) - success_count

        print("\nBackground Removal Summary:")
        print(f"Total images processed: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {failure_count}")

        if failure_count > 0:
            print("\nFailed Images:")
            for success, file_path, error in results:
                if not success:
                    print(f"- {file_path}: {error}")