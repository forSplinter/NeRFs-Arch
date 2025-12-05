import torch
import torch.nn as nn
import torch.nn.functional as F

#this code is inspired by the original this one https://github.com/peihaowang/nerf-pytorch/blob/main/models/embedder.py with some modifications for our use case
class VolumeRenderer(nn.Module):
    def __init__(self, act_fn=F.relu, white_bkgd=False, raw_noise_std=0.):
        super().__init__()
        self.raw_noise_std = raw_noise_std
        self.white_bkgd = white_bkgd
        self.act_fn = act_fn
    
    def forward(self, raw, z_vals, rays_d, **kwargs):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)
        dists = dists * torch.linalg.norm(rays_d[..., None, :], ord=2, dim=-1)
        
        rgb = torch.sigmoid(raw[..., :3])
        
        noise = 0.
        raw_noise_std = kwargs.get('raw_noise_std', self.raw_noise_std)
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., -1].shape, device=raw.device) * raw_noise_std
        
        alpha = raw[..., -1] + noise
        alpha = 1. - torch.exp(-self.act_fn(alpha) * dists)
        
        Ts = torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], dim=-1)
        Ts = torch.cumprod(Ts, dim=-1)[..., :-1]
        
        weights = alpha * Ts
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
        depth_map = torch.sum(weights * z_vals, dim=-1, keepdim=True)
        acc_map = torch.sum(weights, dim=-1, keepdim=True)
        depth_map[acc_map <= 1e-10] = 1e10
        disp_map = 1. / torch.max(torch.full_like(depth_map, 1e-10), depth_map / acc_map)
        
        white_bkgd = kwargs.get('white_bkgd', self.white_bkgd)
        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map)
        
        return dict(rgb=rgb_map, disp=disp_map, acc=acc_map, weights=weights, depth=depth_map)

class MipVolumeRenderer(nn.Module):
    """Renderer pour Mip-NeRF - attend des midpoints, pas des fenceposts"""
    
    def __init__(self, act_fn=F.relu, white_bkgd=False, raw_noise_std=0.):
        super().__init__()
        self.raw_noise_std = raw_noise_std
        self.white_bkgd = white_bkgd
        self.act_fn = act_fn
    
    def forward(self, raw, z_mids, rays_d, **kwargs):
        """
        Args:
            raw: (N_rays, N_samples, 4) - RGB + density
            z_mids: (N_rays, N_samples) - midpoints des intervalles (PAS fenceposts)
            rays_d: (N_rays, 3)
        """
        dists = z_mids[..., 1:] - z_mids[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=-1)
        dists = dists * torch.linalg.norm(rays_d[..., None, :], ord=2, dim=-1)
        
        rgb = torch.sigmoid(raw[..., :3])  
        density = raw[..., -1]              
        
        noise = 0.
        raw_noise_std = kwargs.get('raw_noise_std', self.raw_noise_std)
        if raw_noise_std > 0.:
            noise = torch.randn_like(density) * raw_noise_std
        
        # Alpha compositing
        alpha = 1. - torch.exp(-self.act_fn(density + noise) * dists)
        
        # Transmittance
        Ts = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]
        
        weights = alpha * Ts
        
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
        depth_map = torch.sum(weights * z_mids, dim=-1, keepdim=True)
        acc_map = torch.sum(weights, dim=-1, keepdim=True)
        
        depth_map = torch.where(acc_map > 1e-10,depth_map / acc_map,torch.full_like(depth_map, 1e10))
        disp_map = 1. / torch.clamp(depth_map, min=1e-10)
        
        if kwargs.get('white_bkgd', self.white_bkgd):
            rgb_map = rgb_map + (1. - acc_map)
        
        return dict(rgb=rgb_map, disp=disp_map,acc=acc_map,weights=weights,depth=depth_map)