#This file is a custom version of the well known colmap2nerf.py for instant-ngp link below if you want to see the original:
#scripts/colmap2nerf.py · master · Thomas Pickles / instant-ngp-tomography · GitLab

import argparse
import os
import json
import numpy as np
import math
from utils.colmap_utils import read_cameras_binary, read_images_binary

def parse_args():
    parser = argparse.ArgumentParser(description="Convert COLMAP files to nerf format transforms.json")
    
    parser.add_argument("--colmap_path", help="Path to COLMAP sparse/0 directory with .bin files")
    parser.add_argument("--text", help="Path to COLMAP text files directory")
    parser.add_argument("--images_path", required=True, help="Input path to the images.")
    parser.add_argument("--out", default="transforms.json", help="Output path.")
    parser.add_argument("--aabb_scale", default=16, choices=["1", "2", "4", "8", "16", "32", "64", "128"], help="Large scene scale factor.")
    parser.add_argument("--skip_early", default=0, help="Skip this many images from the start.")
    parser.add_argument("--keep_colmap_coords", action="store_true", help="Keep transforms.json in COLMAP's original frame of reference.")
    
    args = parser.parse_args()
    
    # Validation
    if not args.colmap_path and not args.text:
        parser.error("Either --colmap_path or --text must be provided")
    if args.colmap_path and args.text:
        parser.error("Provide only one of --colmap_path or --text")
    
    return args

def read_cameras_text(path):
    """Read cameras from text file"""
    print(f"Looking for cameras.txt at: {os.path.join(path, 'cameras.txt')}")  # ← Ajoutez cette ligne
    print(f"Directory exists: {os.path.exists(path)}")
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
            from collections import namedtuple
            Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
            cameras[camera_id] = Camera(camera_id, model, width, height, params)
    return cameras

def read_images_text(path):
    """Read images from text file"""
    images = {}
    with open(os.path.join(path, "images.txt"), "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            if lines[i].startswith("#") or not lines[i].strip():
                continue
            # First line: image data
            data = lines[i].split()
            image_id = int(data[0])
            qvec = np.array(list(map(float, data[1:5])))
            tvec = np.array(list(map(float, data[5:8])))
            camera_id = int(data[8])
            name = ' '.join(data[9:])  # Handle filenames with spaces
            
            # Second line: 2D points (skip for NeRF)
            from collections import namedtuple
            Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
            images[image_id] = Image(image_id, qvec, tvec, camera_id, name, None, None)
    return images

def qvec2rotmat(qvec):
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

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db):
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def main():
    args = parse_args()
    
    AABB_SCALE = int(args.aabb_scale)
    SKIP_EARLY = int(args.skip_early)
    IMAGE_FOLDER = args.images_path
    OUT_PATH = args.out
    
    print(f"outputting to {OUT_PATH}...")
    
    # Determine input path and read method
    if args.text:
        input_path = args.text
        print("Reading from text files...")
        cameras = read_cameras_text(input_path)
        images = read_images_text(input_path)
    else:
        input_path = args.colmap_path
        print("Reading from binary files...")
        cameras = read_cameras_binary(os.path.join(input_path, "cameras.bin"))
        images = read_images_binary(os.path.join(input_path, "images.bin"))
    
    if not cameras:
        print("Error: No cameras found in COLMAP reconstruction")
        return
    
    # Get camera parameters from first camera
    cam = list(cameras.values())[0]
    w = float(cam.width)
    h = float(cam.height)
    
    # Extract parameters based on camera model
    if cam.model == "SIMPLE_RADIAL":
        fl_x = float(cam.params[0])
        fl_y = float(cam.params[0])
        cx = float(cam.params[1])
        cy = float(cam.params[2])
        k1 = float(cam.params[3])
        k2 = 0.0
        p1 = 0.0
        p2 = 0.0
    elif cam.model == "PINHOLE":
        fl_x = float(cam.params[0])
        fl_y = float(cam.params[1])
        cx = float(cam.params[2])
        cy = float(cam.params[3])
        k1 = 0.0
        k2 = 0.0
        p1 = 0.0
        p2 = 0.0
    elif cam.model == "OPENCV":
        fl_x = float(cam.params[0])
        fl_y = float(cam.params[1])
        cx = float(cam.params[2])
        cy = float(cam.params[3])
        k1 = float(cam.params[4])
        k2 = float(cam.params[5])
        p1 = float(cam.params[6])
        p2 = float(cam.params[7])
    else:
        print(f"Warning: Camera model {cam.model} not fully supported, using defaults")
        fl_x = float(cam.params[0])
        fl_y = float(cam.params[0])
        cx = w / 2
        cy = h / 2
        k1 = 0.0
        k2 = 0.0
        p1 = 0.0
        p2 = 0.0
    
    is_fisheye = False
    angle_x = math.atan(w / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2
    fovx = angle_x * 180 / math.pi
    fovy = angle_y * 180 / math.pi

    print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")

    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    out = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "is_fisheye": is_fisheye,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "aabb_scale": AABB_SCALE,
        "frames": [],
    }

    up = np.zeros(3)
    frames_added = 0
    
    for i, (image_id, image) in enumerate(images.items()):
        if i < SKIP_EARLY:
            continue
            
        # Use relative path for images
        image_name = image.name
        image_rel_path = os.path.relpath(os.path.join(IMAGE_FOLDER, image_name))
        
        qvec = image.qvec
        tvec = image.tvec
        
        R = qvec2rotmat(qvec)
        t = tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(m)
        
        if not args.keep_colmap_coords:
            c2w[0:3, 2] *= -1
            c2w[0:3, 1] *= -1
            c2w = c2w[[1, 0, 2, 3], :]
            c2w[2, :] *= -1
            up += c2w[0:3, 1]

        frame = {
            "file_path": f"./{image_rel_path}",
            "transform_matrix": c2w
        }
        out["frames"].append(frame)
        frames_added += 1

    nframes = len(out["frames"])

    if args.keep_colmap_coords:
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat)
    else:
        up = up / np.linalg.norm(up)
        print("up vector was", up)
        R = rotmat(up, [0, 0, 1])
        R = np.pad(R, [0, 1])
        R[-1, -1] = 1

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"])

        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in out["frames"]:
            mf = f["transform_matrix"][0:3, :]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3, :]
                p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
                if w > 0.00001:
                    totp += p * w
                    totw += w
        if totw > 0.0:
            totp /= totw
        print(totp)
        
        for f in out["frames"]:
            f["transform_matrix"][0:3, 3] -= totp

        avglen = 0.
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        
        for f in out["frames"]:
            f["transform_matrix"][0:3, 3] *= 4.0 / avglen

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
        
    print(nframes, "frames")
    print(f"writing {OUT_PATH}")
    
    os.makedirs(os.path.dirname(OUT_PATH) if os.path.dirname(OUT_PATH) else ".", exist_ok=True)
    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)

    print("done.")

if __name__ == "__main__":
    main()