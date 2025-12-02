
import os
import json
import numpy as np
import math
from collections import namedtuple
from typing import Dict, List, Tuple, Optional

from utils.colmap.colmap_utils import read_cameras_binary, read_images_binary

# Définition locale pour éviter les conflits d'import
Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

class ColmapToNeRFConverter:
    """Convertisseur COLMAP vers format NeRF"""
    
    @staticmethod
    def read_cameras_text(path: str) -> Dict[int, Camera]:
        """Lecture des caméras depuis fichier texte COLMAP"""
        cameras = {}
        with open(os.path.join(path, "cameras.txt"), "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                data = line.split()
                camera_id = int(data[0])
                model = data[1]
                width = int(data[2])
                height = int(data[3])
                params = list(map(float, data[4:]))
                cameras[camera_id] = Camera(camera_id, model, width, height, params)
        return cameras
    
    @staticmethod
    def read_images_text(path: str) -> Dict[int, Image]:
        """Lecture des images depuis fichier texte COLMAP"""
        images = {}
        with open(os.path.join(path, "images.txt"), "r") as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                if lines[i].startswith("#") or not lines[i].strip():
                    continue
                data = lines[i].split()
                image_id = int(data[0])
                qvec = np.array(list(map(float, data[1:5])))
                tvec = np.array(list(map(float, data[5:8])))
                camera_id = int(data[8])
                name = ' '.join(data[9:])
                images[image_id] = Image(image_id, qvec, tvec, camera_id, name, None, None)
        return images
    
    @staticmethod
    def _qvec_to_rotation_matrix(qvec: np.ndarray) -> np.ndarray:
        """Conversion quaternion vers matrice de rotation"""
        return np.array([
            [
                1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
            ], [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
            ], [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
            ]
        ])
    @staticmethod
    def _rotation_matrix_align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrice de rotation alignant le vecteur a sur b"""
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if c < -1 + 1e-10:
            return ColmapToNeRFConverter._rotation_matrix_align_vectors(
                a + np.random.uniform(-1e-2, 1e-2, 3), b
            )
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))
    
    @staticmethod
    def _closest_point_between_lines(origin_a: np.ndarray, direction_a: np.ndarray,
                                     origin_b: np.ndarray, direction_b: np.ndarray) -> Tuple[np.ndarray, float]:
        """Point le plus proche entre deux lignes"""
        direction_a = direction_a / np.linalg.norm(direction_a)
        direction_b = direction_b / np.linalg.norm(direction_b)
        cross_product = np.cross(direction_a, direction_b)
        denom = np.linalg.norm(cross_product)**2
        translation = origin_b - origin_a
        t_a = np.linalg.det([translation, direction_b, cross_product]) / (denom + 1e-10)
        t_b = np.linalg.det([translation, direction_a, cross_product]) / (denom + 1e-10)
        if t_a > 0:
            t_a = 0
        if t_b > 0:
            t_b = 0
        return (origin_a + t_a * direction_a + origin_b + t_b * direction_b) * 0.5, float(denom)
    
    @staticmethod
    def _extract_camera_parameters(camera) -> Tuple:
        """Extraction des paramètres caméra depuis modèle COLMAP"""
        width = float(camera.width)
        height = float(camera.height)
        
        if camera.model == "SIMPLE_RADIAL":
            focal_x = focal_y = float(camera.params[0])
            center_x, center_y = float(camera.params[1]), float(camera.params[2])
            distortion_k1, distortion_k2 = float(camera.params[3]), 0.0
            distortion_p1, distortion_p2 = 0.0, 0.0
        elif camera.model == "PINHOLE":
            focal_x, focal_y = float(camera.params[0]), float(camera.params[1])
            center_x, center_y = float(camera.params[2]), float(camera.params[3])
            distortion_k1 = distortion_k2 = distortion_p1 = distortion_p2 = 0.0
        elif camera.model == "OPENCV":
            focal_x, focal_y = float(camera.params[0]), float(camera.params[1])
            center_x, center_y = float(camera.params[2]), float(camera.params[3])
            distortion_k1, distortion_k2 = float(camera.params[4]), float(camera.params[5])
            distortion_p1, distortion_p2 = float(camera.params[6]), float(camera.params[7])
        else:
            focal_x = focal_y = float(camera.params[0])
            center_x, center_y = width / 2, height / 2
            distortion_k1 = distortion_k2 = distortion_p1 = distortion_p2 = 0.0
        
        return (width, height, focal_x, focal_y, center_x, center_y, 
                distortion_k1, distortion_k2, distortion_p1, distortion_p2)
    
    @staticmethod
    def _calculate_field_of_view(focal_x: float, focal_y: float, 
                                 width: float, height: float) -> Tuple[float, float]:
        """Calcul du champ de vision à partir de la focale et résolution"""
        angle_x = math.atan(width / (focal_x * 2)) * 2
        angle_y = math.atan(height / (focal_y * 2)) * 2
        return angle_x, angle_y
    
    @staticmethod
    def _process_image_frame(image: Image, images_directory: str, 
                            up_vector: np.ndarray, preserve_colmap_coordinates: bool) -> Tuple[Dict, np.ndarray]:
        """Traitement d'une image unique"""
        image_relative_path = os.path.relpath(os.path.join(images_directory, image.name))
        
        rotation_matrix = ColmapToNeRFConverter._qvec_to_rotation_matrix(image.qvec)
        translation_vector = image.tvec.reshape([3, 1])
        homogeneous_row = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        
        model_matrix = np.concatenate([np.concatenate([rotation_matrix, translation_vector], 1), homogeneous_row], 0)
        camera_to_world = np.linalg.inv(model_matrix)
        
        if not preserve_colmap_coordinates:
            camera_to_world[0:3, 2] *= -1
            camera_to_world[0:3, 1] *= -1
            camera_to_world = camera_to_world[[1, 0, 2, 3], :]
            camera_to_world[2, :] *= -1
            up_vector += camera_to_world[0:3, 1]
        
        return {
            "file_path": f"./{image_relative_path}",
            "transform_matrix": camera_to_world
        }, up_vector
    
    @staticmethod
    def _normalize_coordinate_frames(frames: List[Dict], preserve_colmap_coordinates: bool) -> List[Dict]:
        """Normalisation des systèmes de coordonnées et centrage de la scène"""
        if preserve_colmap_coordinates:
            flip_transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            for frame in frames:
                frame["transform_matrix"] = np.matmul(frame["transform_matrix"], flip_transform)
            return frames
        
        up_vector = np.zeros(3)
        for frame in frames:
            up_vector += frame["transform_matrix"][0:3, 1]
        up_vector = up_vector / np.linalg.norm(up_vector)
        
        alignment_rotation = ColmapToNeRFConverter._rotation_matrix_align_vectors(up_vector, np.array([0, 0, 1]))
        alignment_rotation = np.pad(alignment_rotation, [0, 1])
        alignment_rotation[-1, -1] = 1
        
        for frame in frames:
            frame["transform_matrix"] = np.matmul(alignment_rotation, frame["transform_matrix"])
        
        # Calcul du centre d'attention
        total_weight = 0.0
        center_point = np.array([0.0, 0.0, 0.0])
        
        for frame_a in frames:
            matrix_a = frame_a["transform_matrix"][0:3, :]
            for frame_b in frames:
                matrix_b = frame_b["transform_matrix"][0:3, :]
                point, weight = ColmapToNeRFConverter._closest_point_between_lines(
                    matrix_a[:, 3], matrix_a[:, 2], matrix_b[:, 3], matrix_b[:, 2]
                )
                if weight > 0.00001:
                    center_point += point * weight
                    total_weight += weight
        
        if total_weight > 0.0:
            center_point /= total_weight
        
        # Centrage et mise à l'échelle
        for frame in frames:
            frame["transform_matrix"][0:3, 3] -= center_point
        
        average_distance = sum(np.linalg.norm(frame["transform_matrix"][0:3, 3]) for frame in frames) / len(frames)
        scaling_factor = 4.0 / average_distance
        
        for frame in frames:
            frame["transform_matrix"][0:3, 3] *= scaling_factor
        
        return frames
    
    @staticmethod
    def _convert_image_type(image):
        """Convertit une image de n'importe quel format vers le format local"""
        # Si c'est déjà le bon type
        if hasattr(image, 'qvec') and hasattr(image, 'tvec'):
            return Image(
                id=getattr(image, 'id', 0),
                qvec=getattr(image, 'qvec', np.zeros(4)),
                tvec=getattr(image, 'tvec', np.zeros(3)),
                camera_id=getattr(image, 'camera_id', 0),
                name=getattr(image, 'name', ''),
                xys=getattr(image, 'xys', None),
                point3D_ids=getattr(image, 'point3D_ids', None)
            )
        return image
    
    @staticmethod
    def convert(colmap_binary_path: Optional[str] = None,
                colmap_text_path: Optional[str] = None,
                images_directory: Optional[str] = None,
                output_file: str = "transforms.json",
                aabb_scale: int = 16,
                skip_initial_images: int = 0,
                preserve_colmap_coordinates: bool = False) -> bool:
        """Conversion principale COLMAP vers format NeRF"""
        
        # Lecture des données COLMAP
        if colmap_text_path:
            cameras = ColmapToNeRFConverter.read_cameras_text(colmap_text_path)
            images = ColmapToNeRFConverter.read_images_text(colmap_text_path)
        elif colmap_binary_path:
            cameras = read_cameras_binary(os.path.join(colmap_binary_path, "cameras.bin"))
            images = read_images_binary(os.path.join(colmap_binary_path, "images.bin"))
        else:
            raise ValueError("Either colmap_text_path or colmap_binary_path must be provided")
        
        if not cameras:
            return False
        
        # Extraction paramètres caméra
        camera = list(cameras.values())[0]
        (width, height, focal_x, focal_y, center_x, center_y,
         distortion_k1, distortion_k2, distortion_p1, distortion_p2) = ColmapToNeRFConverter._extract_camera_parameters(camera)
        
        field_of_view_x, field_of_view_y = ColmapToNeRFConverter._calculate_field_of_view(focal_x, focal_y, width, height)
        
        output_structure = {
            "camera_angle_x": field_of_view_x,
            "camera_angle_y": field_of_view_y,
            "fl_x": focal_x, "fl_y": focal_y,
            "k1": distortion_k1, "k2": distortion_k2,
            "p1": distortion_p1, "p2": distortion_p2,
            "is_fisheye": False,
            "cx": center_x, "cy": center_y,
            "w": width, "h": height,
            "aabb_scale": aabb_scale,
            "frames": [],
        }
        
        processed_frames = []
        up_vector = np.zeros(3)
        
        for index, (image_id, image) in enumerate(images.items()):
            if index < skip_initial_images:
                continue
            
            # Conversion du type d'image si nécessaire
            converted_image = ColmapToNeRFConverter._convert_image_type(image)
            
            frame, up_vector = ColmapToNeRFConverter._process_image_frame(
                converted_image, images_directory or ".", up_vector, preserve_colmap_coordinates
            )
            processed_frames.append(frame)
        
        # Normalisation des coordonnées
        processed_frames = ColmapToNeRFConverter._normalize_coordinate_frames(
            processed_frames, preserve_colmap_coordinates
        )
        
        # Conversion des arrays numpy en listes
        for frame in processed_frames:
            frame["transform_matrix"] = frame["transform_matrix"].tolist()
        
        output_structure["frames"] = processed_frames
        
        # Écriture du fichier de sortie
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, "w") as file:
            json.dump(output_structure, file, indent=2)
        
        return True