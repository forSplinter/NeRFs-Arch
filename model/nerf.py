# models/nerf.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from utils.nerfmlp import NerfMLP
from utils.volumerendering import VolumeRenderer
from utils.sampler import StratifiedSampler, HierarchicalSampler


class NeRF(nn.Module):
    def __init__(self, net_depth: int = 8, net_width: int = 256, net_depth_fine: int = 8, net_width_fine: int = 256,N_samples: int = 64,
        N_importance: int = 128, use_viewdirs: bool = True, use_embed: bool = True, multires: int = 10, multires_views: int = 4,
        ray_chunk: int = 1024 * 32, pts_chunk: int = 1024 * 64, perturb: float = 1.0, raw_noise_std: float = 0.0, white_bkgd: bool = False, 
        use_hierarchical: bool = True):
        """_summary_

        Args:
            net_depth (int, optional): _description_. Defaults to 8.
            net_width (int, optional): _description_. Defaults to 256.
            net_depth_fine (int, optional): _description_. Defaults to 8.
            net_width_fine (int, optional): _description_. Defaults to 256.
            N_samples (int, optional): _description_. Defaults to 64.
            N_importance (int, optional): _description_. Defaults to 128.
            use_viewdirs (bool, optional): _description_. Defaults to True.
            use_embed (bool, optional): _description_. Defaults to True.
            multires (int, optional): _description_. Defaults to 10.
            multires_views (int, optional): _description_. Defaults to 4.
            ray_chunk (int, optional): _description_. Defaults to 1024*32.
            pts_chunk (int, optional): _description_. Defaults to 1024*64.
            perturb (float, optional): _description_. Defaults to 1.0.
            raw_noise_std (float, optional): _description_. Defaults to 0.0.
            white_bkgd (bool, optional): _description_. Defaults to False.
            use_hierarchical (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        
        self.N_samples = N_samples
        self.N_importance = N_importance
        self.use_viewdirs = use_viewdirs
        self.ray_chunk = ray_chunk
        self.pts_chunk = pts_chunk
        self.perturb = perturb
        self.use_hierarchical = use_hierarchical
        
        self.nerf_coarse = NerfMLP( input_dim=3, output_dim=4,net_depth=net_depth,net_width=net_width,
            viewdirs=use_viewdirs,use_embed=use_embed, multires=multires, multires_views=multires_views,netchunk=pts_chunk
        )
        
        if use_hierarchical:
            self.nerf_fine = NerfMLP( input_dim=3, output_dim=4, net_depth=net_depth_fine, net_width=net_width_fine,
                viewdirs=use_viewdirs, use_embed=use_embed, multires=multires,multires_views=multires_views,netchunk=pts_chunk
            )
        else:
            self.nerf_fine = None
        
        # Renderer
        self.renderer = VolumeRenderer(
            raw_noise_std=raw_noise_std,
            white_bkgd=white_bkgd
        )
        
        # Samplers
        self.sampler_coarse = StratifiedSampler(
            N_samples=N_samples,
            perturb=perturb,
            lindisp=False
        )
        
        if use_hierarchical:
            self.sampler_fine = HierarchicalSampler(
                N_importance=N_importance,
                perturb=perturb,
                lindisp=False
            )
        else:
            self.sampler_fine = None
    
    def render_rays( self, rays_o: torch.Tensor, rays_d: torch.Tensor, bounds: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Volumetric rendering of rays.
        
        Args:
            rays_o: (N_rays, 3) ray origins
            rays_d: (N_rays, 3) ray directions
            bounds: (N_rays, 2) [near, far] bounds
            
        Returns:
            Dictionary with rendered outputs
        """
        device = rays_o.device
        pts_coarse, z_vals_coarse = self.sampler_coarse(
            rays_o, rays_d, bounds, **kwargs
        )

        # Expand view directions
        viewdirs = rays_d[..., None, :].expand(pts_coarse.shape) if self.use_viewdirs else None
        # Coarse network forward
        raw_coarse = self.nerf_coarse(pts_coarse, viewdirs)
        # Render coarse
        outputs_coarse = self.renderer(
            raw_coarse,
            z_vals_coarse[..., :-1],  # Remove last fencepost
            rays_d,
            **kwargs
        )
        # Hierarchical sampling
        if self.use_hierarchical and self.nerf_fine is not None:
            # Blur weights for stability
            weights = outputs_coarse['weights']
            weights_pad = torch.cat([
                weights[..., :1],
                weights,
                weights[..., -1:]
            ], dim=-1)
            weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
            weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])
            
            z_mids = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
            pts_fine, z_vals_fine, extras = self.sampler_fine(
                rays_o, rays_d, z_mids, weights_blur, **kwargs
            )
            
            viewdirs_fine = rays_d[..., None, :].expand(pts_fine.shape) if self.use_viewdirs else None
            # Fine network forward
            raw_fine = self.nerf_fine(pts_fine, viewdirs_fine)
            # Render fine
            outputs_fine = self.renderer(raw_fine, z_vals_fine[..., :-1], rays_d,**kwargs)
            # Combine outputs
            outputs = outputs_fine
            # Add coarse outputs with '0' suffix
            for key in outputs_coarse:
                outputs[key + '0'] = outputs_coarse[key]
            
            # Add z_std (standard deviation of fine samples)
            outputs['z_std'] = torch.std(extras['z_samples'], dim=-1, unbiased=False)
        else:
            outputs = outputs_coarse
        
        return outputs
    
    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor, bounds: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with ray chunking.
        
        Args:
            rays_o: (..., 3) ray origins
            rays_d: (..., 3) ray directions
            bounds: (..., 2) [near, far] bounds
            
        Returns:
            Dictionary with rendered outputs
        """
        input_shape = rays_o.shape[:-1]
        
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        bounds_flat = bounds.reshape(-1, 2)
        
        rays_d_flat = rays_d_flat / torch.norm(rays_d_flat, dim=-1, keepdim=True)
        
        all_outputs = {}
        for i in range(0, rays_o_flat.shape[0], self.ray_chunk):
            end = min(i + self.ray_chunk, rays_o_flat.shape[0])
            
            chunk_outputs = self.render_rays( rays_o_flat[i:end], rays_d_flat[i:end], bounds_flat[i:end], **kwargs)
            
            for key, val in chunk_outputs.items():
                if key not in all_outputs:
                    all_outputs[key] = []
                all_outputs[key].append(val)
        
        for key in all_outputs:
            all_outputs[key] = torch.cat(all_outputs[key], dim=0)
        
        for key in all_outputs:
            val_shape = all_outputs[key].shape[1:]
            output_shape = tuple(input_shape) + val_shape
            all_outputs[key] = all_outputs[key].reshape(output_shape)
        
        return all_outputs