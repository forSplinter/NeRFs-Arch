# models/mipnerf.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from utils.nerfmlp import MipMLP
from utils.volumerendering import MipVolumeRenderer
from utils.sampler import StratifiedSampler, HierarchicalSampler


class MipNeRF(nn.Module):
    def __init__(self, net_depth: int = 8, net_width: int = 256, net_depth_fine: int = 8, net_width_fine: int = 256,N_samples: int = 64,
        N_importance: int = 128, use_viewdirs: bool = True, use_embed: bool = True,multires: int = 10, multires_views: int = 4,
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
        
        self.nerf_coarse = MipMLP( input_dim=3, output_dim=4,net_depth=net_depth,net_width=net_width,
            viewdirs=use_viewdirs,use_embed=use_embed, multires=multires, multires_views=multires_views,netchunk=pts_chunk
        )
        
        if use_hierarchical:
            self.nerf_fine = MipMLP( input_dim=3, output_dim=4, net_depth=net_depth_fine, net_width=net_width_fine,
                viewdirs=use_viewdirs, use_embed=use_embed, multires=multires,multires_views=multires_views,netchunk=pts_chunk
            )
        else:
            self.nerf_fine = None
        
        # Renderer
        self.renderer = MipVolumeRenderer(
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
    
    def lift_gaussian(self, rays_d: torch.Tensor, t_mean: torch.Tensor, t_var: torch.Tensor, r_var: torch.Tensor, diag: bool = True
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            rays_d (torch.Tensor): _description_
            t_mean (torch.Tensor): _description_
            t_var (torch.Tensor): _description_
            r_var (torch.Tensor): _description_
            diag (bool, optional): _description_. Defaults to True.

        Raises:
            NotImplementedError: _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        # Mean: rays_d * t_mean
        mean = rays_d[..., None, :] * t_mean[..., None]
        
        # Magnitude squared of direction
        d_mag_sq = torch.maximum(
            torch.full_like(rays_d[..., :1], 1e-10),
            torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        )
        
        if diag:
            # Diagonal covariance computation
            d_outer_diag = rays_d ** 2  # (N_rays, 3)
            null_outer_diag = 1.0 - d_outer_diag / d_mag_sq  # (N_rays, 3)
            # Variance along ray direction
            t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]  # (N_rays, N_samples, 3)
            # Variance perpendicular to ray
            xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]  # (N_rays, N_samples, 3)
            # Total covariance
            cov_diag = t_cov_diag + xy_cov_diag
            
            return mean, cov_diag
        else:
            raise NotImplementedError("Full covariance not implemented")
    
    def conical_frustum_to_gaussian( self, rays_d: torch.Tensor,t0: torch.Tensor, t1: torch.Tensor, base_radius: torch.Tensor, diag: bool = True,stable: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Approximate a conical frustum as a Gaussian distribution.
        
        Args:
            rays_d: (N_rays, 3) ray directions
            t0: (N_rays, N_samples) starting distances
            t1: (N_rays, N_samples) ending distances
            base_radius: (N_rays, N_samples) base radius at distance=1
            diag: diagonal covariance only
            stable: use stable computation
            
        Returns:
            mean: (N_rays, N_samples, 3) Gaussian means
            cov: (N_rays, N_samples, 3) Gaussian covariances
        """
        if stable:
            # Stable formulation from Mip-NeRF paper
            mu = (t0 + t1) / 2.0
            hw = (t1 - t0) / 2.0
            
            t_mean = mu + (2.0 * mu * hw ** 2) / (3.0 * mu ** 2 + hw ** 2)
            
            t_var = (hw ** 2) / 3.0 - (4.0 / 15.0) * (
                (hw ** 4 * (12.0 * mu ** 2 - hw ** 2)) /
                (3.0 * mu ** 2 + hw ** 2) ** 2
            )
            
            r_var = base_radius ** 2 * (
                (mu ** 2) / 4.0 +
                (5.0 / 12.0) * hw ** 2 -
                (4.0 / 15.0) * (hw ** 4) / (3.0 * mu ** 2 + hw ** 2)
            )
        else:
            # Unstable formulation (original)
            t_mean = (3.0 * (t1 ** 4 - t0 ** 4)) / (4.0 * (t1 ** 3 - t0 ** 3))
            
            r_var = base_radius ** 2 * (
                3.0 / 20.0 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
            )
            
            t_mosq = 3.0 / 5.0 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
            t_var = t_mosq - t_mean ** 2
        
        return self.lift_gaussian(rays_d, t_mean, t_var, r_var, diag)
    
    def cast_rays( self, z_vals: torch.Tensor, rays_o: torch.Tensor, rays_d: torch.Tensor, radii: torch.Tensor,ray_shape: str = 'cone',diag: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cast rays and convert to Gaussians.
        
        Args:
            z_vals: (N_rays, N_samples+1) sampled distances (fenceposts)
            rays_o: (N_rays, 3) ray origins
            rays_d: (N_rays, 3) ray directions
            radii: (N_rays,) ray radii (pixel size)
            ray_shape: 'cone' or 'cylinder'
            diag: diagonal covariance only
            
        Returns:
            means: (N_rays, N_samples, 3) Gaussian means
            covs: (N_rays, N_samples, 3) Gaussian covariances
        """
        t0 = z_vals[..., :-1] 
        t1 = z_vals[..., 1:]   
        
        # Expand radii to match samples
        radii = radii[..., None].expand_as(t0)
        
        if ray_shape == 'cone':
            means, covs = self.conical_frustum_to_gaussian(
                rays_d, t0, t1, radii, diag=diag
            )
        elif ray_shape == 'cylinder':
            # Cylinder approximation (simpler, not commonly used)
            t_mean = (t0 + t1) / 2.0
            r_var = radii ** 2 / 4.0
            t_var = (t1 - t0) ** 2 / 12.0
            means, covs = self.lift_gaussian(rays_d, t_mean, t_var, r_var, diag)
        else:
            raise ValueError(f"Unknown ray shape: {ray_shape}")
        
        # Add ray origin
        means = means + rays_o[..., None, :]
        
        return means, covs
    
    def render_rays( self, rays_o: torch.Tensor, rays_d: torch.Tensor,bounds: torch.Tensor, radii: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Volumetric rendering of rays.
        
        Args:
            rays_o: (N_rays, 3) ray origins
            rays_d: (N_rays, 3) ray directions
            bounds: (N_rays, 2) [near, far] bounds
            radii: (N_rays,) ray radii
            
        Returns:
            Dictionary with rendered outputs
        """
        device = rays_o.device
        z_vals_coarse, _ = self.sampler_coarse(rays_o, rays_d, bounds, zvals_only=True, **kwargs)
        pts_coarse, cov_coarse = self.cast_rays(z_vals_coarse, rays_o, rays_d, radii)
        z_mids_coarse = 0.5 * (z_vals_coarse[..., :-1] + z_vals_coarse[..., 1:])
        viewdirs = rays_d[..., None, :].expand(pts_coarse.shape) if self.use_viewdirs else None
        raw_coarse = self.nerf_coarse(pts_coarse, cov_coarse, viewdirs)
        # Render coarse
        outputs_coarse = self.renderer(raw_coarse, z_mids_coarse, rays_d, **kwargs)

        if self.use_hierarchical and self.nerf_fine is not None and self.sampler_fine is not None:
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
            z_vals_fine, _, extras = self.sampler_fine(
                rays_o, rays_d, z_mids, weights_blur, zvals_only=True, **kwargs
            )
            pts_fine, cov_fine = self.cast_rays(
                z_vals_fine, rays_o, rays_d, radii
            )
            
            viewdirs_fine = rays_d[..., None, :].expand(pts_fine.shape) if self.use_viewdirs else None
            # Fine network forward
            raw_fine = self.nerf_fine(pts_fine, cov_fine, viewdirs_fine)
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
    
    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor, bounds: torch.Tensor, radii: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with ray chunking.
        
        Args:
            rays_o: (..., 3) ray origins
            rays_d: (..., 3) ray directions
            bounds: (..., 2) [near, far] bounds
            radii: (...,) ray radii
            
        Returns:
            Dictionary with rendered outputs
        """
        input_shape = rays_o.shape[:-1]
        
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        bounds_flat = bounds.reshape(-1, 2)
        radii_flat = radii.reshape(-1)
        
        rays_d_flat = rays_d_flat / torch.norm(rays_d_flat, dim=-1, keepdim=True)
        
        all_outputs = {}
        for i in range(0, rays_o_flat.shape[0], self.ray_chunk):
            end = min(i + self.ray_chunk, rays_o_flat.shape[0])
            
            chunk_outputs = self.render_rays( rays_o_flat[i:end], rays_d_flat[i:end], bounds_flat[i:end],radii_flat[i:end],**kwargs)
            
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