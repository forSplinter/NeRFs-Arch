import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Callable
import math


# γ(p) = (sin(2^0 π p), cos(2^0 π p), ..., sin(2^(L-1) π p), cos(2^(L-1) π p))

class PositionalEncoding(nn.Module):
    def __init__(self, L: int, include_input: bool = True, trainable: bool = False,log_sampling: bool = True, periodic_fn: Optional[List[Callable]] = None) -> None:
        """_summary_

        Args:
            L (int): _description_
            include_input (bool, optional): _description_. Defaults to True.
            trainable (bool, optional): _description_. Defaults to False.
            log_sampling (bool, optional): _description_. Defaults to True.
            periodic_fn (Optional[List[Callable]], optional): _description_. Defaults to None.
        """
        super().__init__()
        self.L = L
        self.include_input = include_input
        self.trainable = trainable
        
        if periodic_fn is None:
            self.periodic_fn = [torch.sin, torch.cos]
        else:
            self.periodic_fn = periodic_fn
        
        if log_sampling:
            freq = 2.0 ** torch.linspace(0.0, self.L - 1, steps=self.L)
        else:
            freq = torch.linspace(2.0 ** 0.0, 2.0 ** (self.L - 1), steps=self.L)
        
        if trainable:
            self.freq = nn.Parameter(freq, requires_grad=True)
        else:
            self.register_buffer("freq", freq)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            p (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        pe = []
        for fn in self.periodic_fn:
            p_freq = p[..., :, None] * self.freq * math.pi  # [..., input_dim, L]
            encoded = fn(p_freq)  # [..., input_dim, L]
            encoded = encoded.transpose(-1, -2)  # [..., L, input_dim]
            pe.append(encoded)
        
        pe = torch.stack(pe, dim=-2)  # [..., L, num_fns, input_dim]
        pe = pe.reshape(p.shape[:-1] + (-1,))  # [..., L * num_fns * input_dim]
        
        if self.include_input:
            pe = torch.cat([p, pe], dim=-1)
        
        return pe
    

class IntegratedPositionalEncoding(nn.Module):
    def __init__(self, L: int, num_freqs: int, include_input: bool = True, log_sampling: bool = True, trainable: bool = False):
        """_summary_

        Args:
            L (int): _description_
            num_freqs (int): _description_
            include_input (bool, optional): _description_. Defaults to True.
            log_sampling (bool, optional): _description_. Defaults to True.
            trainable (bool, optional): _description_. Defaults to False.
        """
        
        super().__init__()
        self.L = L
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = log_sampling

        if log_sampling:
            freq = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        else:
            freq = torch.linspace(2.0 ** 0.0, 2.0 ** (num_freqs - 1), num_freqs)

        if trainable:
            self.freq = nn.Parameter(freq, requires_grad=True)
        else:
            self.register_buffer("freq", freq)
        
    def expected_sin(self, x: torch.Tensor, x_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            x_var (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        y = torch.exp(-0.5 * x_var) * torch.sin(x)
        y_var = torch.clamp(
            0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y ** 2, 
            min=0.0
        )
        return y, y_var
    
    def forward(self, mu: torch.Tensor, cov: torch.Tensor, diag: bool = True) -> torch.Tensor:
        """_summary_

        Args:
            mu (torch.Tensor): _description_
            cov (torch.Tensor): _description_
            diag (bool, optional): _description_. Defaults to True.

        Returns:
            torch.Tensor: _description_
        """
        if not diag:
            cov = torch.diagonal(cov, dim1=-2, dim2=-1)  # [..., input_dim]
        
        # Scale by squared frequencies
        y = mu[..., None, :] * self.freq[:, None]
        y = y.reshape(*mu.shape[:-1], -1)

        y_var = cov[..., None, :] * (self.freq[:, None] ** 2)
        y_var = y_var.reshape(*cov.shape[:-1], -1)
        
        # Sine encoding
        sin_in = y 
        sin_var = y_var
        sin_enc, _ = self.expected_sin(sin_in, sin_var)
        
        # Cosine encoding (shifted by π/2)
        cos_in = y + 0.5 * math.pi
        cos_var = y_var 
        cos_enc, _ = self.expected_sin(cos_in, cos_var)
        
        pe = torch.cat([sin_enc, cos_enc], dim=-1)

        if self.include_input:
            pe = torch.cat([mu, pe], dim=-1)
        
        return pe


class HashEncoding(nn.Module):
    def __init__(self, L: int, F: int, input_dim: int, include_input: bool = True, trainable: bool = False, 
                 base_resolution: int = 16,per_level_scale: float = 1.3819,log2_hashmap_size: int = 19):
        """_summary_

        Args:
            L (int): _description_
            F (int): _description_
            input_dim (int): _description_
            include_input (bool, optional): _description_. Defaults to True.
            trainable (bool, optional): _description_. Defaults to False.
            base_resolution (int, optional): _description_. Defaults to 16.
            per_level_scale (float, optional): _description_. Defaults to 1.3819.
            log2_hashmap_size (int, optional): _description_. Defaults to 19.
        """
        super().__init__()
        self.L = L
        self.F = F
        self.input_dim = input_dim
        self.include_input = include_input
        self.trainable = trainable
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        self.hashmap_size = 2 ** log2_hashmap_size
        
        # Create hash table for each level
        self.embeddings = nn.ParameterList([
            nn.Parameter(
                torch.randn(self.hashmap_size, F) * 0.01, 
                requires_grad=trainable
            ) for _ in range(L)
        ])
    
    def hash_fn(self, coords: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            coords (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        primes = torch.tensor(
            [1540171, 1721003, 1890971], 
            device=coords.device,
            dtype=coords.dtype
        )
        idx = (coords * primes).sum(dim=-1).long()
        return idx % self.hashmap_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) or (N, 3) positions in [0, 1]
        
        Returns:
            (B, N, encoded_dim) or (N, encoded_dim) encoded features
        """
        # Handle both (B, N, 3) and (N, 3) inputs
        input_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (N, 3) -> (1, N, 3)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, N, _ = x.shape
        out = []

        for l in range(self.L):
            resolution = self.base_resolution * (self.per_level_scale ** l)
            scaled_x = x * resolution 
            idx0 = torch.floor(scaled_x).long()
            idx1 = idx0 + 1
            frac = scaled_x - idx0.float()
        
            # Get 8 corners of the voxel
            corners = [
                torch.stack([idx0[..., 0], idx0[..., 1], idx0[..., 2]], dim=-1),
                torch.stack([idx1[..., 0], idx0[..., 1], idx0[..., 2]], dim=-1),
                torch.stack([idx0[..., 0], idx1[..., 1], idx0[..., 2]], dim=-1),
                torch.stack([idx1[..., 0], idx1[..., 1], idx0[..., 2]], dim=-1),
                torch.stack([idx0[..., 0], idx0[..., 1], idx1[..., 2]], dim=-1),
                torch.stack([idx1[..., 0], idx0[..., 1], idx1[..., 2]], dim=-1),
                torch.stack([idx0[..., 0], idx1[..., 1], idx1[..., 2]], dim=-1),
                torch.stack([idx1[..., 0], idx1[..., 1], idx1[..., 2]], dim=-1),
            ]

            # Lookup features at each corner
            corner_feats = [self.embeddings[l][self.hash_fn(corner)] for corner in corners]
            
            # Trilinear interpolation weights
            fx, fy, fz = frac[..., 0:1], frac[..., 1:2], frac[..., 2:3]
            weights = [
                (1 - fx) * (1 - fy) * (1 - fz),
                fx * (1 - fy) * (1 - fz),
                (1 - fx) * fy * (1 - fz),
                fx * fy * (1 - fz),
                (1 - fx) * (1 - fy) * fz,
                fx * (1 - fy) * fz,
                (1 - fx) * fy * fz,
                fx * fy * fz,
            ]
            
            # Interpolate
            level_feat = sum(w * f for w, f in zip(weights, corner_feats))
            out.append(level_feat)
        
        pe = torch.cat(out, dim=-1)
        
        if self.include_input:
            pe = torch.cat([x, pe], dim=-1)
        
        # Restore original shape if needed
        if squeeze_output:
            pe = pe.squeeze(0)  # (1, N, encoded_dim) -> (N, encoded_dim)
        
        return pe

class TriangularEncoding(nn.Module):
    def __init__(self, L: int, include_input: bool = True):
        pass 
    def forward(self, x: torch.Tensor):
       pass 

class OneBlobEncoding(nn.Module):
    def __init__(self, include_input: bool = True):
        pass 
    def forward(self, x: torch.Tensor):
       pass