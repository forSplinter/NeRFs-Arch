import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from tqdm import tqdm
import json
import time
from utils.metrics import Metrics, img2mse, mse2psnr
from utils.export import Exporter
from utils.camera.camera import Camera


class Eval:
    def __init__( self, model: nn.Module, dataset: torch.utils.data.Dataset, device: str = 'cuda',output_dir: Optional[str] = None):
        """_summary_

        Args:
            model (nn.Module): _description_
            dataset (torch.utils.data.Dataset): _description_
            device (str, optional): _description_. Defaults to 'cuda'.
            output_dir (Optional[str], optional): _description_. Defaults to None.
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        
        self.output_dir = Path(output_dir) if output_dir else Path('eval_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize your metrics and exporter
        self.metrics = Metrics(device=device)
        self.exporter = Exporter()
        
        # Results storage
        self.results = {
            'per_image': [],
            'aggregate': {}
        }
    
    @torch.no_grad()
    def evaluate_single_image(self, rays_o: torch.Tensor, rays_d: torch.Tensor, bounds: torch.Tensor, radii: torch.Tensor, target_rgb: torch.Tensor,H: int, W: int,**kwargs) -> Dict:
        """_summary_

        Args:
            rays_o (torch.Tensor): _description_
            rays_d (torch.Tensor): _description_
            bounds (torch.Tensor): _description_
            radii (torch.Tensor): _description_
            target_rgb (torch.Tensor): _description_
            H (int): _description_
            W (int): _description_

        Returns:
            Dict: _description_
        """
        self.model.eval()
        
        # Render image
        outputs = self.model(rays_o, rays_d, bounds, radii, **kwargs)
        
        # Get RGB output (fine if available, else coarse)
        if 'rgb' in outputs:
            pred_rgb = outputs['rgb']
        else:
            pred_rgb = outputs.get('rgb0', outputs['rgb'])
        
        # Reshape to image
        pred_rgb = pred_rgb.reshape(H, W, 3)
        target_rgb = target_rgb.reshape(H, W, 3)
        
        # Compute all metrics using your Metrics class
        metrics_dict = self.metrics.compute_all(pred_rgb, target_rgb, format='HWC')
        
        # Convert tensor values to float
        metrics_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                       for k, v in metrics_dict.items()}
        
        return {
            'pred_rgb': pred_rgb,
            'target_rgb': target_rgb,
            'metrics': metrics_dict,
            'depth': outputs.get('depth', None),
            'acc': outputs.get('acc', None)
        }
    
    @torch.no_grad()
    def evaluate_dataset( self, save_images: bool = True, save_comparison: bool = True, verbose: bool = True,**render_kwargs) -> Dict:
        """
        Evaluate entire dataset
        
        Args:
            save_images: Save rendered images using Exporter
            save_comparison: Save side-by-side comparisons
            verbose: Print progress
            render_kwargs: Additional rendering arguments
            
        Returns:
            Dictionary with aggregate metrics
        """
        self.model.eval()
        
        all_metrics = {
            'mse': [],
            'psnr': [],
            'ssim': [],
            'lpips': []
        }
        
        rendered_images = []
        target_images = []
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Evaluating {len(self.dataset)} images")
            print(f"{'='*80}\n")
        
        pbar = tqdm(range(len(self.dataset)), desc="Evaluating") if verbose else range(len(self.dataset))
        
        for idx in pbar:
            # Get data from your RayNeRFDataset
            data = self.dataset[idx]
            
            rays = data['rays']  # (2, H*W, 3) or (H*W, 2, 3)
            target_rgb = data['target_s']  # (H*W, 3)
            
            # Extract rays_o and rays_d
            if rays.shape[0] == 2:
                rays_o, rays_d = rays[0], rays[1]
            else:
                rays_o, rays_d = rays[:, 0], rays[:, 1]
            
            # Get dimensions
            H, W = self.dataset.height_width()
            
            # Get bounds
            near, far = self.dataset.near_far()
            bounds = torch.tensor(
                [[near, far]], device=self.device
            ).expand(rays_o.shape[0], 2)
            
            # Get radii (pixel size)
            radii = torch.full((rays_o.shape[0],), self.dataset.radii(), device=self.device)
            
            # Evaluate
            result = self.evaluate_single_image(
                rays_o, rays_d, bounds, radii, target_rgb, H, W, **render_kwargs
            )
            
            # Store metrics
            for key in all_metrics:
                all_metrics[key].append(result['metrics'][key])
            
            # Store per-image result
            self.results['per_image'].append({
                'index': idx,
                'metrics': result['metrics']
            })
            
            rendered_images.append(result['pred_rgb'])
            target_images.append(result['target_rgb'])
            
            if verbose:
                pbar.set_postfix({
                    'PSNR': f"{result['metrics']['psnr']:.2f}",
                    'SSIM': f"{result['metrics']['ssim']:.4f}"
                })
        
        if save_images:
            img_dir = self.output_dir / 'rendered'
            print(f"\nSaving rendered images to {img_dir}...")
            self.exporter.save_images(
                rendered_images,
                output_dir=img_dir,
                prefix='pred',
                verbose=False
            )
            
            if save_comparison:
                target_dir = self.output_dir / 'ground_truth'
                print(f"Saving ground truth images to {target_dir}...")
                self.exporter.save_images(
                    target_images,
                    output_dir=target_dir,
                    prefix='target',
                    verbose=False
                )
                
                comp_dir = self.output_dir / 'comparisons'
                comp_dir.mkdir(exist_ok=True)
                print(f"Saving comparison images to {comp_dir}...")
                for i, (pred, target) in enumerate(zip(rendered_images, target_images)):
                    self._save_comparison(pred, target, comp_dir / f'comp_{i:04d}.png')

        aggregate_metrics = {}
        for key in all_metrics:
            values = torch.tensor(all_metrics[key])
            aggregate_metrics[key] = {
                'mean': values.mean().item(),
                'std': values.std().item(),
                'min': values.min().item(),
                'max': values.max().item()
            }
        
        self.results['aggregate'] = aggregate_metrics
        
        # Save results
        self._save_results()
        
        if verbose:
            self._print_summary()
        
        return aggregate_metrics
    
    def _save_comparison( self, pred: torch.Tensor, target: torch.Tensor,path: Path):
        import matplotlib.pyplot as plt
        
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        error = torch.abs(pred - target).cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(pred_np)
        axes[0].set_title('Predicted')
        axes[0].axis('off')
        
        axes[1].imshow(target_np)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        im = axes[2].imshow(error, cmap='hot', vmin=0, vmax=0.5)
        axes[2].set_title('Error')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """Save evaluation results to JSON"""
        results_path = self.output_dir / 'results.json'
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
    
    def _print_summary(self):
        """Print evaluation summary"""
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}\n")
        
        agg = self.results['aggregate']
        
        print(f"{'Metric':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print(f"{'-'*80}")
        
        print(f"{'PSNR (dB)':<10} "
              f"{agg['psnr']['mean']:>10.2f}  "
              f"{agg['psnr']['std']:>10.2f}  "
              f"{agg['psnr']['min']:>10.2f}  "
              f"{agg['psnr']['max']:>10.2f}")
        
        print(f"{'SSIM':<10} "
              f"{agg['ssim']['mean']:>10.4f}  "
              f"{agg['ssim']['std']:>10.4f}  "
              f"{agg['ssim']['min']:>10.4f}  "
              f"{agg['ssim']['max']:>10.4f}")
        
        print(f"{'LPIPS':<10} "
              f"{agg['lpips']['mean']:>10.4f}  "
              f"{agg['lpips']['std']:>10.4f}  "
              f"{agg['lpips']['min']:>10.4f}  "
              f"{agg['lpips']['max']:>10.4f}")
        
        print(f"\n{'='*80}\n")
    
    @torch.no_grad()
    def render_novel_view( self, camera: Camera, H: int, W: int, near: float, far: float, save_path: Optional[Path] = None, **render_kwargs) -> torch.Tensor:
        """_summary_

        Args:
            camera (Camera): _description_
            H (int): _description_
            W (int): _description_
            near (float): _description_
            far (float): _description_
            save_path (Optional[Path], optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """
        self.model.eval()
        
        rays_o, rays_d = camera.rays(H, W, device=self.device)
        bounds = torch.tensor([[near, far]], device=self.device).expand(rays_o.shape[0], 2)
        
        focal = camera.intrinsics.fl_x
        radii = torch.full((rays_o.shape[0],), 2.0 / focal, device=self.device)
        outputs = self.model(rays_o, rays_d, bounds, radii, **render_kwargs)
        rgb = outputs.get('rgb', outputs.get('rgb0'))
        rgb = rgb.reshape(H, W, 3)
        
        if save_path:
            self.exporter.save_image(rgb, save_path)
            print(f"Novel view saved to {save_path}")
        
        return rgb
    
    @torch.no_grad()
    def render_video( self, cameras: List[Camera], H: int, W: int, near: float, far: float, output_path: Path, fps: int = 30, quality: int = 8, **render_kwargs):
        frames = []
        
        print(f"\nRendering video with {len(cameras)} frames...")
        
        for i, camera in enumerate(tqdm(cameras, desc="Rendering")):
            frame = self.render_novel_view(
                camera, H, W, near, far, **render_kwargs
            )
            frames.append(frame.cpu())
            
            if (i + 1) % 10 == 0:
                print(f"  Rendered {i + 1}/{len(cameras)} frames")
        
        self.exporter.save_video( frames, output_path=output_path, fps=fps,quality=quality)
