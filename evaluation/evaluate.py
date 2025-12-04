# utils/evaluation.py

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, List
from pathlib import Path
from tqdm import tqdm
import json

from utils.metrics import Metrics
from utils.export import Exporter
from utils.camera.camera import Camera


class Eval:
    """
    Evaluateur optimisé pour NeRF
    """
    
    def __init__(self, model: nn.Module, dataset, device: str = 'cuda', output_dir: Optional[str] = None):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.eval() 
        
        self.output_dir = Path(output_dir) if output_dir else Path('eval_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = Metrics(device=device)
        self.exporter = Exporter()
        
        self.results = {
            'per_image': [],
            'aggregate': {}
        }
    
    @torch.no_grad()
    def evaluate_single(self, idx: int, **render_kwargs) -> Dict:
        """
        Évaluer une seule image
        """
        data = self.dataset[idx]
        
        rays = data['rays']
        if rays.shape[0] == 2:
            rays_o, rays_d = rays[0].to(self.device), rays[1].to(self.device)
        else:
            rays_o, rays_d = rays[:, 0].to(self.device), rays[:, 1].to(self.device)
        
        H, W = self.dataset.H, self.dataset.W
        near, far = self.dataset.near, self.dataset.far
        
        num_rays = rays_o.shape[0]
        bounds = torch.tensor([[near, far]], device=self.device).expand(num_rays, 2)
        radii = torch.full((num_rays,), 2.0 / self.dataset.focal_x, device=self.device)
        
        outputs = self.model(rays_o, rays_d, bounds, radii, **render_kwargs)
        pred_rgb = outputs.get('rgb_fine', outputs.get('rgb_coarse'))
        if pred_rgb is None:
            pred_rgb = outputs.get('rgb', outputs.get('rgb0'))
        
        pred_rgb = pred_rgb.reshape(H, W, 3)
        
        target_rgb = data.get('target_s', None)
        if target_rgb is not None:
            if len(target_rgb.shape) == 3:  # (H, W, 3)
                target_flat = target_rgb.reshape(-1, 3)
                target_rgb_reshaped = target_rgb
            else:  
                target_flat = target_rgb
                target_rgb_reshaped = target_rgb.reshape(H, W, 3)
            
            metrics_dict = self.metrics.compute_all(pred_rgb, target_rgb_reshaped, format='HWC')
            metrics_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                           for k, v in metrics_dict.items()}
        else:
            metrics_dict = {}
            target_flat = None
            target_rgb_reshaped = None
        
        return {
            'pred_rgb': pred_rgb,
            'target_rgb': target_rgb_reshaped,
            'metrics': metrics_dict,
            'depth': outputs.get('depth', None),
            'acc': outputs.get('acc', None)
        }
    
    @torch.no_grad()
    def evaluate_all(self, 
                    max_images: Optional[int] = None,
                    save_every: int = 5,
                    **render_kwargs) -> Dict:
        """
        Évaluer toutes les images du dataset
        
        Args:
            max_images: Maximum d'images à évaluer (None = toutes)
            save_every: Sauvegarder une image toutes les N images
            render_kwargs: Arguments pour le rendu
        
        Returns:
            Métriques agrégées
        """
        self.model.eval()
        
        total_images = len(self.dataset)
        if max_images is not None:
            total_images = min(total_images, max_images)
        
        print(f"\n🔍 Évaluation de {total_images} images...")
        
        all_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
        
        for idx in tqdm(range(total_images), desc="Images"):
            result = self.evaluate_single(idx, **render_kwargs)
            
            if save_every > 0 and idx % save_every == 0 and result['pred_rgb'] is not None:
                img_path = self.output_dir / f'pred_{idx:04d}.png'
                self.exporter.save_image(result['pred_rgb'], img_path)
                
                if result['target_rgb'] is not None:
                    target_path = self.output_dir / f'target_{idx:04d}.png'
                    self.exporter.save_image(result['target_rgb'], target_path)
            
            if result['metrics']:
                for key in all_metrics:
                    if key in result['metrics']:
                        all_metrics[key].append(result['metrics'][key])
            
            self.results['per_image'].append({
                'index': idx,
                'metrics': result['metrics']
            })
        
        aggregate = {}
        for key, values in all_metrics.items():
            if values:
                values_tensor = torch.tensor(values)
                aggregate[key] = {
                    'mean': values_tensor.mean().item(),
                    'std': values_tensor.std().item(),
                    'min': values_tensor.min().item(),
                    'max': values_tensor.max().item()
                }
        
        self.results['aggregate'] = aggregate
        
        self._save_results()
        
        if aggregate:
            self._print_summary(aggregate)
        
        return aggregate
    
    def _save_results(self):
        """Sauvegarder résultats en JSON"""
        def convert(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().tolist() if obj.numel() > 1 else obj.cpu().item()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            else:
                return obj
        
        json_path = self.output_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(convert(self.results), f, indent=2)
        
        print (f"Result saved: {json_path}")
    
    def _print_summary(self, aggregate: Dict):
        print(f"\n{'='*50}")
        print("Results")
        
        if 'psnr' in aggregate:
            psnr = aggregate['psnr']
            print(f"PSNR: {psnr['mean']:.2f} ± {psnr['std']:.2f} dB")
            print(f"     [min: {psnr['min']:.2f}, max: {psnr['max']:.2f}]")
        
        if 'ssim' in aggregate:
            ssim = aggregate['ssim']
            print(f"SSIM: {ssim['mean']:.4f} ± {ssim['std']:.4f}")
            print(f"     [min: {ssim['min']:.4f}, max: {ssim['max']:.4f}]")
        
        if 'lpips' in aggregate:
            lpips = aggregate['lpips']
            print(f"LPIPS: {lpips['mean']:.4f} ± {lpips['std']:.4f}")
            print(f"      [min: {lpips['min']:.4f}, max: {lpips['max']:.4f}]")
        
        print(f"{'='*50}\n")
    
    @torch.no_grad()
    def render_video(self, 
                    cameras: List[Camera],
                    H: int,
                    W: int,
                    output_path: str,
                    fps: int = 30,
                    **render_kwargs):
      
        frames = []
        print(f"\n🎬 video render ({len(cameras)} frames)...")
        
        for i, camera in enumerate(tqdm(cameras, desc="Render")):
            rays_o, rays_d = camera.rays(H, W, device=self.device)
            
            num_rays = rays_o.shape[0]
            near, far = self.dataset.near, self.dataset.far
            bounds = torch.tensor([[near, far]], device=self.device).expand(num_rays, 2)
            radii = torch.full((num_rays,), 2.0 / camera.intrinsics.fl_x, device=self.device)
            
            outputs = self.model(rays_o, rays_d, bounds, radii, **render_kwargs)
            
            rgb = outputs.get('rgb_fine', outputs.get('rgb_coarse'))
            if rgb is None:
                rgb = outputs.get('rgb', outputs.get('rgb0'))
            
            if rgb is not None:
                rgb = rgb.reshape(H, W, 3).cpu()
                frames.append(rgb)
        
        if frames:
            self.exporter.save_video(frames, output_path=output_path, fps=fps)
            print(f"Video saved: {output_path}")