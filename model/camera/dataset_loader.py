import json 
import os
from .camera_intrinsics import CameraIntrinsics
from .camera_extrinsics import CameraExtrinsics
from .camera import Camera
import numpy as np


class DatasetLoader:
    def __init__(self, json_path):
        self.data = self._load_json(json_path)
        self.intrinsics = self._parse_intrinsics(self.data)
        self.frames = self._parse_frames(self.data)

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _parse_intrinsics(self, data):
        return CameraIntrinsics(
            fl_x=data["fl_x"],
            fl_y=data["fl_y"],
            cx=data["cx"],
            cy=data["cy"],
            w=int(data["w"]),
            h=int(data["h"]),
            k1=data.get("k1", 0.0),
            k2=data.get("k2", 0.0),
            p1=data.get("p1", 0.0),
            p2=data.get("p2", 0.0),
            is_fisheye=data.get("is_fisheye", False),
        )

    def _parse_frames(self, data):
        frames = []
        for frame_data in data["frames"]:
            transform = np.array(frame_data["transform_matrix"])
            extrinsics = CameraExtrinsics(transform_matrix=transform)
            frame = Camera(
                intrinsics=self.intrinsics,
                extrinsics=extrinsics,
                image_path=frame_data["file_path"]
            )
            frames.append(frame)
        return frames

                
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> dict:
        return self.frames[idx]