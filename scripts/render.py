#!/usr/bin/env python3
"""
Render an orbit video around a NeRF scene.
Usage:
    python scripts/render_orbit.py --config configs/lot1.yaml --ckpt logs/lot1_exp/checkpoints/00040000.ckpt
"""

from pathlib import Path
import argparse
import yaml
import torch
import numpy as np
import math
from tqdm import tqdm

from model.mip_nerf import MipNeRF
from utils.export import Exporter


def parse_args():
    parser = argparse.ArgumentParser(description="Render orbit video from NeRF checkpoint")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint (default: latest)')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--num_frames', type=int, default=120, help='Number of frames in orbit')
    parser.add_argument('--fps', type=int, default=30, help='FPS for output video')
    parser.add_argument('--height', type=int, default=512, help='Output image height')
    parser.add_argument('--width', type=int, default=512, help='Output image width')
    parser.add_argument('--radius', type=float, default=4.5, help='Camera orbit radius')
    parser.add_argument('--elevation', type=float, default=0.3, help='Camera elevation (radians)')
    parser.add_argument('--chunk', type=int, default=4096, help='Chunk size for rendering')
    parser.add_argument('--focal', type=float, default=None, help='Focal length (default: auto)')
    parser.add_argument('--center', type=float, nargs=3, default=[0, 0, 0], help='Scene center point')
    return parser.parse_args()


def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path, device, model_config, data_config):
    """Load model from checkpoint"""
    
    model_kwargs = {
        'net_depth': model_config.get('net_depth', 8),
        'net_width': model_config.get('net_width', 256),
        'net_depth_fine': model_config.get('net_depth_fine', model_config.get('net_depth', 8)),
        'net_width_fine': model_config.get('net_width_fine', model_config.get('net_width', 256)),
        'N_samples': model_config.get('N_samples', 64),
        'N_importance': model_config.get('N_importance', 128),
        'use_viewdirs': model_config.get('use_viewdirs', True),
        'use_embed': model_config.get('use_embed', True),
        'multires': model_config.get('multires', 10),
        'multires_views': model_config.get('multires_views', 4),
        'ray_chunk': model_config.get('ray_chunk', 1024 * 32),
        'pts_chunk': model_config.get('pts_chunk', 1024 * 64),
        'perturb': 0.0,
        'raw_noise_std': 0.0,
        'white_bkgd': data_config.get('white_bkgd', False),
        'use_hierarchical': model_config.get('use_hierarchical', True)
    }
    
    model = MipNeRF(**model_kwargs)
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    state_dict = ckpt['model']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    step = ckpt.get('step', 0)
    return model, step


def find_checkpoint(exp_dir, ckpt_path=None):
    """Find checkpoint to load"""
    ckpt_dir = exp_dir / 'checkpoints'
    
    if ckpt_path:
        return Path(ckpt_path)
    
    if (ckpt_dir / 'final.ckpt').exists():
        return ckpt_dir / 'final.ckpt'
    
    ckpt_files = sorted(ckpt_dir.glob('*.ckpt'))
    if ckpt_files:
        return max(ckpt_files, key=lambda x: x.stat().st_mtime)
    
    raise ValueError(f"No checkpoint found in {ckpt_dir}")


def create_orbit_poses(num_frames, radius, elevation, center):
    """
    Create camera poses for an orbit around the scene center.
    
    Args:
        num_frames: Number of frames in the orbit
        radius: Distance from center
        elevation: Camera elevation angle in radians
        center: Scene center point [x, y, z]
    
    Returns:
        List of 4x4 camera-to-world matrices
    """
    poses = []
    center = np.array(center)
    
    for i in range(num_frames):
        # Angle around the orbit
        theta = 2 * math.pi * i / num_frames
        
        # Camera position
        x = radius * math.cos(theta) * math.cos(elevation)
        y = radius * math.sin(theta) * math.cos(elevation)
        z = radius * math.sin(elevation)
        
        camera_pos = np.array([x, y, z]) + center
        
        forward = center - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0, 0, 1])
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward  # Camera looks along -Z
        c2w[:3, 3] = camera_pos
        
        poses.append(c2w)
    
    return poses


def get_rays_from_pose(pose, H, W, focal, device):
    """
    Generate rays from a camera pose.
    
    Args:
        pose: 4x4 camera-to-world matrix
        H, W: Image dimensions
        focal: Focal length
        device: torch device
    
    Returns:
        rays_o: (H*W, 3) ray origins
        rays_d: (H*W, 3) ray directions
    """
    # Pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing='xy'
    )
    
    # Camera coordinates (OpenGL convention: -Z is forward)
    cx, cy = W / 2, H / 2
    dirs = torch.stack([
        (i - cx) / focal,
        -(j - cy) / focal,  # Flip Y
        -torch.ones_like(i)  # -Z direction
    ], dim=-1)  # (H, W, 3)
    
    # Convert pose to tensor
    pose = torch.tensor(pose, dtype=torch.float32, device=device)
    
    # Transform to world coordinates
    rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)  # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    # Ray origins (camera position)
    rays_o = pose[:3, 3].expand(H, W, 3)  # (H, W, 3)
    
    # Flatten
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    return rays_o, rays_d


def render_frame(model, pose, H, W, focal, near, far, device, chunk_size):
    """Render a single frame from a camera pose"""
    
    rays_o, rays_d = get_rays_from_pose(pose, H, W, focal, device)
    
    num_rays = rays_o.shape[0]
    bounds = torch.tensor([[near, far]], device=device).expand(num_rays, 2)
    
    # Compute radii for mip-nerf
    radii = 2.0 / max(H, W) * 2 / math.sqrt(12)
    radii_tensor = torch.full((num_rays,), radii, device=device)
    
    all_rgb = []
    
    with torch.no_grad():
        for i in range(0, num_rays, chunk_size):
            end = min(i + chunk_size, num_rays)
            outputs = model(
                rays_o[i:end],
                rays_d[i:end],
                bounds[i:end],
                radii_tensor[i:end]
            )
            rgb = outputs.get('rgb', outputs.get('rgb0'))
            all_rgb.append(rgb.cpu())
    
    rgb = torch.cat(all_rgb, dim=0).reshape(H, W, 3)
    return rgb


def main():
    args = parse_args()
    
    config = load_config(args.config)
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup paths
    exp_name = config['experiment_name']
    basedir = config.get('basedir', './logs')
    exp_dir = Path(basedir) / exp_name
    
    output_dir = Path(args.output) if args.output else exp_dir / 'orbit_video'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    ckpt_path = find_checkpoint(exp_dir, args.ckpt)
    print(f"Loading checkpoint: {ckpt_path}")
    
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    model, step = load_model(ckpt_path, device, model_config, data_config)
    print(f"Loaded model from step {step}")
    
    # Parameters
    near = data_config.get('near', 2.0)
    far = data_config.get('far', 8.0)
    H, W = args.height, args.width
    focal = args.focal or (W / 2)  # Default focal length
    
    print(f"Rendering {args.num_frames} frames at {W}x{H}")
    print(f"Orbit radius: {args.radius}, elevation: {args.elevation}")
    print(f"Near/Far: [{near}, {far}]")
    
    # Generate orbit poses
    poses = create_orbit_poses(
        args.num_frames,
        args.radius,
        args.elevation,
        args.center
    )
    
    # Render frames
    exporter = Exporter()
    frames = []
    
    for i, pose in enumerate(tqdm(poses, desc="Rendering orbit")):
        rgb = render_frame(
            model, pose, H, W, focal,
            near, far, device, args.chunk
        )
        frames.append(rgb)
        
        # Save individual frame
        exporter.save_image(rgb, output_dir / f'frame_{i:04d}.png')
    
    # Create video
    print("Creating video...")
    video_path = output_dir / f'orbit_step{step}.mp4'
    exporter.save_video(frames, video_path, fps=args.fps)
    
    print(f"\nRendering complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - {args.num_frames} frames (frame_*.png)")
    print(f"  - Video: {video_path}")


if __name__ == '__main__':
    main()