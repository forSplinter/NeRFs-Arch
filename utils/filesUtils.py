import os
import shutil
import glob
from pathlib import Path


def create_dir_if_not_exists(base_dir: str, folders: list):
    """Create a directory if it does not exist.

    Args:
        base_dir (str): Path to the base directory.
    """

    for foler in folders:
        path = os.path.join(base_dir, foler)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")


def copy_files(
    src_dir: str,
    dest_dir: str,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"),
):
    os.makedirs(dest_dir, exist_ok=True)
    copied_files = []
    for ext in extensions:
        pattern = os.path.join(src_dir, f"*{ext}")
        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path)
            dst_path = os.path.join(dest_dir, filename)
            shutil.copy2(file_path, dst_path)
            copied_files.append(dst_path)

    print(f"Copied {len(copied_files)} files from {src_dir} to {dest_dir}")
    return copied_files


def get_sub(parent_dir: str) -> list:
    """_summary_

    Args:
        parent_dir (str): _description_

    Returns:
        list: _description_
    """
    return [
        d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))
    ]


def clean_directory(dir_path: str):
    """_summary_

    Args:
        dir_path (str): _description_
    """
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
