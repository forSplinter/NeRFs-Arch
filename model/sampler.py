import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class StratifiedSampler(nn.Module):
    def __init__(self, N_samples: int, perturb: float = 0.0, lindisp: bool = False, pytest: bool = False):
        """_summary_

        Args:
            N_samples (int): _description_
            perturb (float, optional): _description_. Defaults to 0.0.
            lindisp (bool, optional): _description_. Defaults to False.
            pytest (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.N_samples = N_samples
        self.perturb = perturb
        self.lindisp = lindisp
        self.pytest = pytest
         
    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor, bounds: torch.Tensor,zvals_only: bool = False,**render_kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """_summary_

        Args:
            rays_o (torch.Tensor): _description_
            rays_d (torch.Tensor): _description_
            bounds (torch.Tensor): _description_
            zvals_only (bool, optional): _description_. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: _description_
        """
        perturb = render_kwargs.get('perturb', self.perturb)
        N_samples = render_kwargs.get('N_samples', self.N_samples)
        
        N_rays = rays_o.shape[0]
        device = rays_o.device
        
        near, far = bounds[..., 0, None], bounds[..., 1, None]  # [N_rays, 1]
        
        t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
        
        if not self.lindisp:
            z_vals = near * (1.0 - t_vals) + far * t_vals
        else:
            z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
        
        z_vals = z_vals.expand(N_rays, N_samples)  # [N_rays, N_samples]
        
        if perturb > 0.0:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand(z_vals.shape, device=device)
            
            if self.pytest:
                import numpy as np
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand).to(device)
            
            z_vals = lower + (upper - lower) * t_rand
        
        if not zvals_only:
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
            return pts, z_vals
        else:
            return z_vals, None


class HierarchicalSampler(nn.Module):
    def __init__(self, N_importance: int, perturb: float = 0.0, lindisp: bool = False, pytest: bool = False):
        """_summary_

        Args:
            N_importance (int): _description_
            perturb (float, optional): _description_. Defaults to 0.0.
            lindisp (bool, optional): _description_. Defaults to False.
            pytest (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.N_importance = N_importance
        self.perturb = perturb
        self.lindisp = lindisp
        self.pytest = pytest
    
    def sample_pdf( self,  bins: torch.Tensor, weights: torch.Tensor, det: bool = False) -> torch.Tensor:
        """_summary_

        Args:
            bins (torch.Tensor): _description_
            weights (torch.Tensor): _description_
            det (bool, optional): _description_. Defaults to False.

        Returns:
            torch.Tensor: _description_
        """
        weights = weights + 1e-5  
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [N_rays, N-1]
        cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, N-1]
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [N_rays, N]
        
        if det:
            u = torch.linspace(0.0, 1.0, steps=self.N_importance, device=cdf.device)
            u = u.expand(list(cdf.shape[:-1]) + [self.N_importance])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [self.N_importance], device=cdf.device)
        
        if self.pytest:
            import numpy as np
            np.random.seed(0)
            new_shape = list(cdf.shape[:-1]) + [self.N_importance]
            if det:
                u = np.linspace(0.0, 1.0, self.N_importance)
                u = np.broadcast_to(u, new_shape)
            else:
                u = np.random.rand(*new_shape)
            u = torch.Tensor(u).to(cdf.device)
        
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], dim=-1)  # [N_rays, N_importance, 2]
        
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
        
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        
        return samples
    
    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor, z_vals: torch.Tensor, weights: torch.Tensor, zvals_only: bool = False,**render_kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """_summary_

        Args:
            rays_o (torch.Tensor): _description_
            rays_d (torch.Tensor): _description_
            z_vals (torch.Tensor): _description_
            weights (torch.Tensor): _description_
            zvals_only (bool, optional): _description_. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]: _description_
        """
        perturb = render_kwargs.get('perturb', self.perturb)
        
        ret_extras = {}
        
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        
        z_samples = self.sample_pdf( z_vals_mid, weights[..., 1:-1], det=(perturb == 0.0))
        z_samples = z_samples.detach()
        
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        
        ret_extras['z_samples'] = z_samples
        
        if not zvals_only:
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            return pts, z_vals, ret_extras
        else:
            return z_vals, None, ret_extras


def compute_weights(density: torch.Tensor, z_vals: torch.Tensor,rays_d: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        density (torch.Tensor): _description_
        z_vals (torch.Tensor): _description_
        rays_d (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists,torch.full_like(dists[..., :1], 1e10)], dim=-1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    alpha = 1.0 - torch.exp(-F.relu(density) * dists)
    transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]),1.0 - alpha[..., :-1] + 1e-10], dim=-1),dim=-1)
    
    weights = alpha * transmittance
    
    return weights